"""E073 (test-time adaptation 1): style-neighbour smoothing over the test set.
Idea: the hidden test has ~4,000 passages (~220 per model, 2-3 per prompt). Passages by the same model on different
prompts should share style, so mixing each passage's blend probabilities with its style neighbours' (excluding its own
estimated topic cluster) could fix errors without any training. Unsupervised and transductive: features come from the
test texts only.
Sim test per draw: one random group of 2-3 answers per prompt for all 150 prompts (~375 rows, ~21 per model; the real
test has ~10x more per model, so this is pessimistic). Blend = members we can score honestly at the 0.664 hand weights,
twins/Gaussian on infer.py's estimated clusters, Sinkhorn 0.45. Style features: words outside the 100 most frequent in the
sim test replaced by '*' (topic masking, as the public 'distort' member), char 2-5 TF-IDF, cosine. Smoothing:
P' = (1 - a) P + a * mean of P over the k nearest neighbours outside the row's own cluster; a in {0, .1, .2, .3, .5}, k in {5, 10}.
Run from the repo root: python experiments/honest/e073_style_graph.py"""
import collections
import glob
import re
import sys

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.preprocessing import normalize

sys.path.insert(0, 'nb/honest_oof')
import gauss as ga  # noqa: E402

sys.path.insert(0, 'paper')
from log import log_row  # noqa: E402

GCFG = dict(red='wpca', d=128, se_shrink='lw', sp_shrink='lw', lam_floor=1e-4, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0)
W = {'q14v5': 3.0, 'q8v5': 2.0, 'll8v5': 1.0}
GAUSS = {'q14v5': 0.70, 'q8v5': 0.44}
cpu = np.load('research/honest_oof/cpu_members_oof.npz', allow_pickle=True)
y, fold = cpu['y'], cpu['fold']
train = pd.read_csv('data/train.csv')
pid, text = train.prompt_id.to_numpy(), train.text.fillna('').astype(str).to_numpy()
G = {}
for f in sorted(glob.glob('research/honest_oof/gpu/honest_oof_*.npz')):
    z = np.load(f)
    G.update({k: z[k] for k in z.files})
prior = np.bincount(y, minlength=18) + 1.0
prior /= prior.sum()


def capped_agglo(S, cap=18, thr=0.15):   # public infer.py
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); k = int(np.argmax(M)); i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr: break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return np.unique(lab, return_inverse=True)[1]


def twin(rc, bc, g):
    mean = np.zeros((g.max() + 1, 18)); np.add.at(mean, g, rc); mean /= np.bincount(g)[:, None]
    return np.where((np.bincount(g)[g] >= 2)[:, None], log_softmax(rc - mean[g] + bc, 1), 0.0)


def blend_probs(rows, g):
    size = np.bincount(g)[g]
    tot, n = np.zeros((len(rows), 18)), 0
    for m, w in W.items():
        tot += w * log_softmax(G[m + '_plain'][rows].astype(float), 1) + 0.35 * w * twin(G[m + '_rc'][rows].astype(float), G[m + '_bc'][rows], g)
        n += 2
        if m in GAUSS:
            out = np.zeros((len(rows), 18))
            for k in range(5):
                sel = (fold[rows] == k) & (size >= 2)
                if sel.any():
                    prm = {'m': G[m + '_m'][k], 'lam': G[m + '_lam'][k], 'T1': 3.0, 'Tg': 3.0, 'cfg': GCFG}
                    out[sel] = ga.scores(None, np.unique(g[sel], return_inverse=True)[1], prm, U=G[m + '_U'][rows[sel]].astype(float))
            tot += GAUSS[m] * out
            n += 1
    tot += 0.25 * cpu['tfidf'][rows] + 0.10 * cpu['distort'][rows] + 12 * cpu['svc4'][rows]
    n += 3
    P = softmax(tot / n, 1)
    tgt = 0.65 * P.mean(0) + 0.35 * prior
    Q = P.copy()
    for _ in range(100):
        Q *= len(Q) * tgt[None] / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
        Q /= Q.sum(1, keepdims=True)
    return softmax(0.55 * np.log(P + 1e-12) + 0.45 * np.log(Q + 1e-12), 1)


WRE = re.compile(r"[A-Za-z’']+")


def style_sim(txt):
    cnt = collections.Counter(w.lower() for t in txt for w in WRE.findall(t))
    keep = {w for w, _ in cnt.most_common(100)}
    masked = [WRE.sub(lambda m: m.group(0) if m.group(0).lower() in keep else '*', t) for t in txt]
    X = normalize(TfidfVectorizer(analyzer='char', ngram_range=(2, 5), sublinear_tf=True, min_df=2).fit_transform(masked))
    return (X @ X.T).toarray()


rng = np.random.default_rng(0)
A, K = (0.0, 0.1, 0.2, 0.3, 0.5), (5, 10)
res, ari, same_model_nn = [], [], []
for d in range(20):
    rows = np.concatenate([rng.choice(np.flatnonzero(pid == p), int(rng.integers(2, 4)), replace=False) for p in np.unique(pid)])
    Xc = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05).fit_transform(text[rows])
    g = capped_agglo((Xc @ Xc.T).toarray())
    P = blend_probs(rows, g)
    S = style_sim(text[rows])
    S[g[:, None] == g[None, :]] = -np.inf          # never a neighbour from the row's own estimated cluster
    order = np.argsort(-S, 1)
    yt = y[rows]
    same_model_nn.append((yt[order[:, 0]] == yt).mean())
    for k in K:
        nb = order[:, :k]
        Pn = P[nb].mean(1)
        for a in A:
            res.append(dict(draw=d, k=k, a=a, bacc=bas(yt, ((1 - a) * P + a * Pn).argmax(1))))
R = pd.DataFrame(res)
Pv = R.pivot_table(index='draw', columns=['k', 'a'], values='bacc')
base = Pv[(5, 0.0)]
rows_out = []
for col in Pv.columns:
    dlt = Pv[col] - base
    rows_out.append(dict(k=col[0], a=col[1], bacc=Pv[col].mean(), diff=dlt.mean(), se=dlt.std() / np.sqrt(len(dlt)), better=f'{(dlt > 0).sum()}/{len(dlt)}'))
S_ = pd.DataFrame(rows_out)
print(f'rows per draw ~{len(rows)}, 20 draws; nearest style neighbour (outside own cluster) is the same model: {np.mean(same_model_nn):.3f} (chance 0.056)')
print(S_.round(4).to_string(index=False))
R.to_csv('experiments/honest/out/e073_style_graph.csv', index=False)
best = S_.iloc[S_['diff'].idxmax()]
log_row(id='E073', category='tta', metric='bacc_sim_1group2-3_per_prompt',
        description='test-time style-neighbour smoothing: mix each passage blend probabilities with its k topic-masked style neighbours (outside its estimated cluster)',
        config='20 draws, one 2-3 group per prompt (150 prompts, ~375 rows); honest blend at 0.664 weights, est. clusters, Sinkhorn 0.45; style = top-100-word masking + char 2-5 TF-IDF on the sim test only; a in {0,.1,.2,.3,.5}, k in {5,10}',
        baseline=f'{base.mean():.4f}', value=f"{best['bacc']:.4f}", folds_pos=best['better'], files='experiments/honest/out/e073_style_graph.csv',
        notes=f"best k={int(best['k'])} a={best['a']} {best['diff']:+.4f} (se {best['se']:.4f}); same-model nearest style neighbour {np.mean(same_model_nn):.3f} vs chance 0.056")
