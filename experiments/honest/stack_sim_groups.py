"""E067: is the Gemma gain in E066 real, or an artefact of true (oracle) prompt groups?
Sim test = prompts of our GroupKFold folds 0-2 (where Gemma has OOF), one random group of 2-3 answers per prompt per
draw. Groups for the centred (twin) and Gaussian heads are either the TRUE groups or ESTIMATED exactly as the public
infer.py does (TF-IDF stop_words english, min_df 2, max_df 0.05; capped average-linkage, cap 18, threshold 0.15).
Configs on the 0.664 hand weights (members we can score honestly): base; existing twins x2 / x4 (control: does any
centred head gain from more weight here?); + Gemma plain / Gemma centred head (recomputed from its saved features and
fold heads for any grouping). Run from the repo root: python experiments/honest/stack_sim_groups.py"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import load_file
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.model_selection import GroupKFold

sys.path.insert(0, 'nb/honest_oof')
import gauss as ga  # noqa: E402

sys.path.insert(0, 'paper')
from log import log_row  # noqa: E402

OUT = Path('experiments/honest/out')
GCFG = dict(red='wpca', d=128, se_shrink='lw', sp_shrink='lw', lam_floor=1e-4, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0)
W0664 = {'q14v5': 3.0, 'q8v5': 2.0, 'll8v5': 1.0, 'tfidf': 0.25, 'distort': 0.10, 'svc4': 12.0}
GAUSS = {'q14v5': 0.70, 'q8v5': 0.44}
TWIN = 0.35
NDRAW = 40

cpu = np.load('research/honest_oof/cpu_members_oof.npz', allow_pickle=True)
y, fold = cpu['y'], cpu['fold']
train = pd.read_csv('data/train.csv')
pid, text = train.prompt_id.to_numpy(), train.text.to_numpy()
G = {}
for f in sorted(glob.glob('research/honest_oof/gpu/honest_oof_*.npz')):
    z = np.load(f)
    G.update({k: z[k] for k in z.files})
LORA = [m for m in ('q14v5', 'q8v5', 'll8v5') if f'{m}_plain' in G and bool(G[f'{m}_split_ok'])]
# Gemma: plain logits, centred-head projection rc = z @ Wc.T and bias, per row from its own fold's heads
ofold = np.full(len(y), -1)
for k, (_, va) in enumerate(GroupKFold(5).split(np.zeros(len(y)), y, pid)):
    ofold[va] = k
GP, GRC, GBC = (np.full((len(y), 18), np.nan) for _ in range(3))
for d in sorted(glob.glob('research/kernels_2026-10-26/gemma_s2/ts/lora_twostage/fold[0-9]')):
    v = np.load(f'{d}/val_outputs.npz', allow_pickle=True)   # our own kernel output
    H = load_file(f'{d}/heads.safetensors')
    r = v['rows']
    GP[r] = v['plain_logits']
    GRC[r] = v['z'].astype(np.float32) @ H['centered.weight'].astype(np.float32).T
    GBC[r] = H['centered.bias']
grow = np.isfinite(GP).all(1)
PROMPTS = np.unique(pid[grow])
print('members', LORA, '+ tfidf/distort/svc4 | Gemma rows', grow.sum(), '| prompts', len(PROMPTS))


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


def twin(rc, bc, gs):
    mean = np.zeros((gs.max() + 1, 18)); np.add.at(mean, gs, rc); mean /= np.bincount(gs)[:, None]
    return np.where((np.bincount(gs)[gs] >= 2)[:, None], log_softmax(rc - mean[gs] + bc, 1), 0.0)


def members(rows, gs):
    size = np.bincount(gs)[gs]
    M = {}
    for m in LORA:
        M[m] = log_softmax(G[m + '_plain'][rows].astype(float), 1)
        M[m + '_twin'] = twin(G[m + '_rc'][rows].astype(float), G[m + '_bc'][rows], gs)
        if m in GAUSS:
            out = np.zeros((len(rows), 18))
            for k in range(5):
                sel = (fold[rows] == k) & (size >= 2)
                # a group can span seeded folds; each row is read with its own fold's head (as the test would be)
                if sel.any():
                    prm = {'m': G[m + '_m'][k], 'lam': G[m + '_lam'][k], 'T1': 3.0, 'Tg': 3.0, 'cfg': GCFG}
                    out[sel] = ga.scores(None, np.unique(gs[sel], return_inverse=True)[1], prm, U=G[m + '_U'][rows[sel]].astype(float))
            M[m + '_gauss'] = out
    for m in ('tfidf', 'distort', 'svc4'):
        M[m] = cpu[m][rows].astype(float)
    M['gemma'] = log_softmax(GP[rows].astype(float), 1)
    M['gemma_twin'] = twin(GRC[rows].astype(float), GBC[rows], gs)
    return M


base = {}
for m in LORA:
    base.update({m: W0664[m], m + '_twin': W0664[m] * TWIN})
    if m in GAUSS:
        base[m + '_gauss'] = GAUSS[m]
base.update({m: W0664[m] for m in ('tfidf', 'distort', 'svc4')})
CONF = {'base': base}
for f in (2, 4):
    CONF[f'twins x{f}'] = {k: v * (f if k.endswith('_twin') else 1) for k, v in base.items()}
for a, b in ((1, 0), (2, 0), (0, 0.7), (0, 1.4), (0, 2.8), (1, 1.4), (1, 2.8), (2, 2.8)):
    CONF[f'gemma plain {a} + centred {b}'] = dict(base, gemma=a, gemma_twin=b)

rng = np.random.default_rng(0)
res = []
ari = []
for d in range(NDRAW):
    rows = np.concatenate([rng.choice(np.flatnonzero(pid == p), int(rng.integers(2, 4)), replace=False) for p in PROMPTS])
    true_g = np.unique(pid[rows], return_inverse=True)[1]
    Xc = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05).fit_transform(text[rows])
    est_g = capped_agglo((Xc @ Xc.T).toarray())
    from sklearn.metrics import adjusted_rand_score
    ari.append(adjusted_rand_score(true_g, est_g))
    for mode, gs in (('true groups', true_g), ('estimated clusters', est_g)):
        M = members(rows, gs)
        for c, w in CONF.items():
            tot = sum(w[k] * M[k] for k in w if w[k] > 0)
            res.append(dict(draw=d, mode=mode, config=c, bacc=bas(y[rows], tot.argmax(1))))
R = pd.DataFrame(res)
piv = R.pivot_table(index=['mode', 'draw'], columns='config', values='bacc')
summ = []
for mode in ('true groups', 'estimated clusters'):
    P = piv.loc[mode]
    for c in CONF:
        dlt = P[c] - P['base']
        summ.append(dict(mode=mode, config=c, bacc=P[c].mean(), diff=dlt.mean(), se=dlt.std() / np.sqrt(len(dlt)), better=f'{(dlt > 0).sum()}/{len(dlt)}'))
S = pd.DataFrame(summ)
print(f'rows per draw ~{len(rows)}, {NDRAW} draws, clustering ARI vs true prompts {np.mean(ari):.3f}')
print(S.round(4).to_string(index=False))
S.to_csv(OUT / 'stack_sim_groups.csv', index=False)
est = S[S['mode'] == 'estimated clusters'].set_index('config')
gem = est.loc[[c for c in est.index if c.startswith('gemma')]]
best = gem['diff'].idxmax()
log_row(id='E067' + ('b' if 'q14v5' in LORA else ''), category='honest-oof', metric='bacc_sim_1group2-3_per_prompt',
        description='Gemma gain check: true groups vs public-infer estimated clusters (TF-IDF capped agglo 0.15) on the honest 0.664 subset; twins upweighted as control',
        config=f'{NDRAW} draws, one 2-3 group per prompt of our folds 0-2 ({len(PROMPTS)} prompts); members ' + '+'.join(LORA) + ' + tfidf/distort/svc4',
        baseline=f"{est.loc['base', 'bacc']:.4f}", value=f"{gem.loc[best, 'bacc']:.4f}", folds_pos=gem.loc[best, 'better'],
        files='experiments/honest/out/stack_sim_groups.csv',
        notes=f"best Gemma config under estimated clusters: {best} {gem.loc[best, 'diff']:+.4f} (se {gem.loc[best, 'se']:.4f}); "
              f"twins x4 est {est.loc['twins x4', 'diff']:+.4f}; ARI {np.mean(ari):.3f}")
