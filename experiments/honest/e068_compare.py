"""E068: do our Kaggle Benchmarks passages (3,333, 12 labels) help the 0.664 CPU members when added to their extra
training file? Compares honest seeded-split OOF from kernel ntl-honest-cpu (E065, without) and ntl-honest-cpu-own
(with): balanced accuracy per member, recall on the 12 served vs the 6 unserved labels, and the paired effect on the
honest blend of the members we can score (0.664 hand weights, test-like 2-3 answer groups, 12 draws).
Run from the repo root after downloading the -own kernel output to research/honest_oof/cpu_own/:
python experiments/honest/e068_compare.py"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score as bas

sys.path.insert(0, 'nb/honest_oof')
import gauss as ga  # noqa: E402

sys.path.insert(0, 'paper')
from log import log_row  # noqa: E402

base = np.load('research/honest_oof/cpu_members_oof.npz', allow_pickle=True)
y, fold, labels = base['y'], base['fold'], list(base['labels'].astype(str))
train = pd.read_csv('data/train.csv')
pid = train.prompt_id.to_numpy()
pos = pd.Series(np.arange(len(y)), index=train.id.astype(str))
own = {}
seen = np.zeros(len(y), int)
for k in range(5):
    z = np.load(f'research/honest_oof/cpu_own/cache/sim{k}_members.npz', allow_pickle=True)   # our kernel output
    r = pos.loc[z['ids'].astype(str)].to_numpy()
    assert (fold[r] == k).all()
    seen[r] += 1
    for m in ('tfidf', 'distort', 'svc4'):
        own.setdefault(m, np.zeros((len(y), 18), np.float32))[r] = z[m]
assert (seen == 1).all()
UNSERVED = {'claude-fable-5', 'gemini-3.7-flash', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3', 'qwen3.8-max'}
uns = np.array([l in UNSERVED for l in labels])
rows = []
for m in ('tfidf', 'distort', 'svc4'):
    for tag, P in (('without', base[m]), ('with own', own[m])):
        p = P.argmax(1)
        rec = np.array([(p[y == i] == i).mean() for i in range(18)])
        rows.append(dict(member=m, data=tag, bacc=bas(y, p), recall_served=rec[~uns].mean(), recall_unserved=rec[uns].mean(),
                         per_fold=[round(bas(y[fold == k], p[fold == k]), 4) for k in range(5)]))
T = pd.DataFrame(rows)
print(T.round(4).to_string(index=False))

# paired blend effect (0.664 hand weights; members we can score honestly)
G = {}
for f in sorted(glob.glob('research/honest_oof/gpu/honest_oof_*.npz')):
    z = np.load(f)
    G.update({k: z[k] for k in z.files})
LORA = ['q14v5', 'q8v5', 'll8v5']
W = {'q14v5': 3.0, 'q8v5': 2.0, 'll8v5': 1.0}
GAUSS = {'q14v5': 0.70, 'q8v5': 0.44}
GCFG = dict(red='wpca', d=128, se_shrink='lw', sp_shrink='lw', lam_floor=1e-4, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0)
rng = np.random.default_rng(0)
res = []
for d in range(12):
    g = np.full(len(y), -1)
    nxt = 0
    for p_ in np.unique(pid):
        ii = rng.permutation(np.flatnonzero(pid == p_))
        i = 0
        while i < len(ii):
            n = 2 if len(ii) - i in (2, 4) else int(rng.integers(2, 4))
            g[ii[i:i + n]] = nxt
            nxt += 1
            i += n
    size = np.bincount(g)[g]
    tot = np.zeros((len(y), 18))
    for m in LORA:
        tot += W[m] * log_softmax(G[m + '_plain'].astype(float), 1)
        rc = G[m + '_rc'].astype(float)
        mean = np.zeros((g.max() + 1, 18)); np.add.at(mean, g, rc); mean /= np.bincount(g)[:, None]
        tot += 0.35 * W[m] * log_softmax(rc - mean[g] + G[m + '_bc'], 1)
        if m in GAUSS:
            out = np.zeros((len(y), 18))
            for k in range(5):
                sel = fold == k
                prm = {'m': G[m + '_m'][k], 'lam': G[m + '_lam'][k], 'T1': 3.0, 'Tg': 3.0, 'cfg': GCFG}
                out[sel] = ga.scores(None, np.unique(g[sel], return_inverse=True)[1], prm, U=G[m + '_U'][sel].astype(float))
            tot += GAUSS[m] * out
    for tag, C in (('without', base), ('with own', own)):
        t = tot + 0.25 * C['tfidf'] + 0.10 * C['distort'] + 12 * C['svc4']
        res.append(dict(draw=d, data=tag, bacc=bas(y, t.argmax(1))))
R = pd.DataFrame(res).pivot(index='draw', columns='data', values='bacc')
dlt = R['with own'] - R['without']
print(f"\nhonest blend: without {R['without'].mean():.4f} | with own {R['with own'].mean():.4f} | diff {dlt.mean():+.4f} "
      f"(se {dlt.std() / np.sqrt(len(dlt)):.4f}, better in {(dlt > 0).sum()}/{len(dlt)} draws)")
T.to_csv('experiments/honest/out/e068_members.csv', index=False)
R.to_csv('experiments/honest/out/e068_blend.csv')
s = T[T.member == 'svc4'].set_index('data')
log_row(id='E068', category='data', metric='bacc_honest_seeded_5fold',
        description='our 3,333 Kaggle Benchmarks passages (12 labels) added to the extra training file of the 0.664 CPU members (tfidf/distort/svc4)',
        config='kernels ntl-honest-cpu (without) vs ntl-honest-cpu-own (with), extra weight 0.5; blend = honest subset, 0.664 hand weights, 12 draws of 2-3 groups',
        baseline=f"{R['without'].mean():.4f}", value=f"{R['with own'].mean():.4f}", folds_pos=f'{(dlt > 0).sum()}/{len(dlt)} draws',
        files='experiments/honest/out/e068_*.csv',
        notes=f"svc4 {s.loc['without', 'bacc']:.4f} -> {s.loc['with own', 'bacc']:.4f}; unserved-label recall "
              f"{s.loc['without', 'recall_unserved']:.3f} -> {s.loc['with own', 'recall_unserved']:.3f}; blend diff {dlt.mean():+.4f}")
