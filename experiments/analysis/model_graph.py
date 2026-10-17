"""Model-family linkage graph (paper analysis). Two independent similarity views between the 18 models:
 (1) confusion affinity: M[a,b] = mean OOF probability of class b over passages of class a (strong blend + Gemma LoRA,
     prompt-disjoint OOF), symmetrised, diagonal removed, rows renormalised;
 (2) style-centroid similarity: frozen Qwen3-8B L7+22 features centred by their TRUE prompt mean (removes topic),
     standardised, per-model centroid, cosine similarity.
Hierarchical clustering (average linkage) on each; prompt-bootstrap (200x) co-clustering frequency at k=6 (number
of labs) and k=9. Writes research/model_graph.md and a dendrogram PNG."""
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import squareform
from scipy.special import log_softmax

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'research'
tr = pd.read_csv(ROOT / 'data/train.csv')
L = sorted(tr.model.unique())
K = len(L)
y = tr.model.map({l: i for i, l in enumerate(L)}).values
g = tr.prompt_id.values
lab = np.array([l.split('-')[0] for l in L])

# ---- view 1: confusion affinity from OOF probabilities
zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
S = sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
cal = lambda Z: log_softmax(Z * minimize_scalar(lambda t: -log_softmax(Z * t, 1)[np.arange(len(y)), y].mean(),
                                                bounds=(0.05, 20), method='bounded').x, 1)
S = cal(S)
G = np.full((len(y), K), np.nan)
pos = {i: k for k, i in enumerate(tr.id.values)}
import glob
for f in glob.glob(str(ROOT / 'nb/lora_gemma/full_out/lora_gemma/fold*/val_outputs.npz')):
    z = np.load(f, allow_pickle=True)
    G[[pos[i] for i in z['ids']]] = z['plain_logits']
P = np.exp(0.5 * S + 0.5 * cal(G))
P /= P.sum(1, keepdims=True)


def confusion_affinity(rows):
    M = np.zeros((K, K))
    for a in range(K):
        r = rows[y[rows] == a]
        M[a] = P[r].mean(0)
    np.fill_diagonal(M, 0)
    M /= M.sum(1, keepdims=True)
    return (M + M.T) / 2


# ---- view 2: style centroids on prompt-centred features
z = np.load(ROOT / 'experiments/papers/qwen3_8b_train_feats.npz')
lays = list(z['layers'])
X = np.concatenate([z['E'][:, lays.index(7)], z['E'][:, lays.index(22)]], 1).astype(np.float64)
Xc = X - pd.DataFrame(X).groupby(g).transform('mean').values
Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)


def centroid_sim(rows):
    C = np.stack([Xc[rows][y[rows] == a].mean(0) for a in range(K)])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C @ C.T


def to_dist(Sim, affinity):
    D = (1 - Sim / Sim.max()) if affinity else (1 - Sim)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    return np.clip(D, 0, None)


allr = np.arange(len(y))
views = {'confusion': (confusion_affinity(allr), True), 'style_centroid': (centroid_sim(allr), False)}
Z = {k: linkage(squareform(to_dist(s, a), checks=False), 'average') for k, (s, a) in views.items()}

# ---- prompt bootstrap co-clustering
rng = np.random.RandomState(0)
up = np.unique(g)
co = {(k, c): np.zeros((K, K)) for k in views for c in (6, 9)}
B = 200
for _ in range(B):
    rows = np.concatenate([np.where(g == p)[0] for p in rng.choice(up, len(up), replace=True)])
    for k, (fn, aff) in {'confusion': (confusion_affinity, True), 'style_centroid': (centroid_sim, False)}.items():
        Zb = linkage(squareform(to_dist(fn(rows), aff), checks=False), 'average')
        for c in (6, 9):
            cl = fcluster(Zb, c, 'maxclust')
            co[(k, c)] += cl[:, None] == cl[None, :]

# ---- report
short = [l.replace('gemini-', 'gem-').replace('claude-', 'cl-').replace('-preview', '') for l in L]
lines = ['# Model-family linkage graph', '',
         'Two views on prompt-disjoint data (train, 150 prompts):',
         '- **confusion**: symmetrised OOF probability mass a->b (strong frozen-LLM blend + Gemma LoRA, 50/50 log-space);',
         '- **style_centroid**: cosine similarity of per-model centroids of frozen Qwen3-8B L7+22 features after removing each',
         '  passage\'s TRUE prompt mean (topic removed).',
         'Clusters: average linkage; stability = share of 200 prompt-bootstrap resamples in which two models share a cluster.', '']
for k, (s, aff) in views.items():
    Sm = s.copy()
    np.fill_diagonal(Sm, np.nan)
    lines += [f'## {k}', '', 'Top-3 nearest models per model:', '']
    for a in range(K):
        nn = np.argsort(-np.nan_to_num(Sm[a], nan=-9))[:3]
        lines.append(f'- **{L[a]}** -> ' + ', '.join(f'{L[b]} ({Sm[a, b]:.3f})' for b in nn))
    for c in (6, 9):
        cl = fcluster(Z[k], c, 'maxclust')
        lines += ['', f'Clusters at k={c} (full data; stability of each pair in brackets):', '']
        for cid in np.unique(cl):
            mem = np.where(cl == cid)[0]
            stab = [co[(k, c)][i, j] / B for i in mem for j in mem if i < j]
            lines.append(f'- {{{", ".join(L[i] for i in mem)}}}' + (f'  [mean pair stability {np.mean(stab):.2f}]' if stab else ''))
    lines.append('')
# cross-lab links: strongest off-lab edges
lines += ['## Strongest cross-lab links (confusion view)', '']
Cf = views['confusion'][0]
pairs = sorted(((Cf[a, b], a, b) for a in range(K) for b in range(a + 1, K) if lab[a] != lab[b]), reverse=True)[:8]
for v, a, b in pairs:
    lines.append(f'- {L[a]} <-> {L[b]}: {v:.3f}')
(OUT / 'model_graph.md').write_text('\n'.join(lines), encoding='utf-8')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for i, k in enumerate(views):
        dendrogram(Z[k], labels=short, orientation='right', ax=ax[i], color_threshold=0)
        ax[i].set_title(k)
    plt.tight_layout()
    plt.savefig(OUT / 'model_graph.png', dpi=130)
    print('png written')
except ImportError:
    print('matplotlib missing: no png')
print('wrote', OUT / 'model_graph.md')
