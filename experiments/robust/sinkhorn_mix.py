"""E053 (test plan #16): robustness of the final Sinkhorn balancing to the test's class mix. On our strong OOF bundle
(research/ext_oof/q35heads reference log-probs, 3 scenarios) draw simulated test sets of 1,000 rows whose class mix
comes from Dirichlet(a) (a = inf: exactly even, as when every prompt is answered by all 18 models; smaller a: more
uneven), then score balanced accuracy after partial balancing log P + s * (log Sinkhorn(P) - log P), s in 0..1
(s = 1 is the full Sinkhorn used by the public stack and v8). CPU only.
Run from the repo root: python experiments/robust/sinkhorn_mix.py"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, 'paper')
from log import LOG, log_row  # noqa: E402

OUT = Path('experiments/robust/out')
OUT.mkdir(parents=True, exist_ok=True)
z = np.load('research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)   # our own OOF bundle
classes = z['classes'].astype(str)
train = pd.read_csv('data/train.csv', dtype=str).set_index('id').loc[z['ids'].astype(str)]
y = np.searchsorted(classes, train.model.to_numpy())
K, N, REPS = 18, 1000, 40
ALPHAS = [np.inf, 10.0, 3.0, 1.0, 0.5]
STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
bacc = lambda yt, p: np.mean([(p[yt == j] == j).mean() for j in np.unique(yt)])


def sinkhorn(P, iters=300):
    Q = P.copy()
    for _ in range(iters):
        Q /= Q.sum(0, keepdims=True)
        Q *= len(Q) / K
        Q /= Q.sum(1, keepdims=True)
    return Q


rng = np.random.default_rng(0)
idx_of = [np.flatnonzero(y == k) for k in range(K)]
rows = []
for s in range(z['reference'].shape[0]):
    logp = z['reference'][s] - np.log(np.exp(z['reference'][s]).sum(1, keepdims=True))
    for a in ALPHAS:
        for r in range(REPS):
            w = np.full(K, 1 / K) if np.isinf(a) else rng.dirichlet(np.full(K, a))
            n_k = np.full(K, N // K) if np.isinf(a) else rng.multinomial(N, w)
            pick = np.concatenate([rng.choice(idx_of[k], n_k[k], replace=True) for k in range(K) if n_k[k]])
            lp, yt = logp[pick], y[pick]
            ls = np.log(np.clip(sinkhorn(np.exp(lp)), 1e-30, None))
            for st in STRENGTHS:
                rows.append(dict(scenario=s, alpha=a, rep=r, strength=st, n_classes=int((n_k > 0).sum()),
                                 max_share=float(n_k.max() / N), bacc=bacc(yt, (lp + st * (ls - lp)).argmax(1))))
df = pd.DataFrame(rows)
df.to_csv(OUT / 'sinkhorn_mix_runs.csv', index=False)
tab = df.groupby(['alpha', 'strength']).bacc.mean().unstack()
base = tab[0.0]
delta = tab.sub(base, axis=0)
delta.to_csv(OUT / 'sinkhorn_mix_delta.csv')
tab.to_csv(OUT / 'sinkhorn_mix_bacc.csv')
win = df.pivot_table(index=['scenario', 'alpha', 'rep'], columns='strength', values='bacc')
share_full_beats_none = (win[1.0] > win[0.0]).groupby(level='alpha').mean()
print('mean balanced accuracy by class-mix concentration (rows) and balancing strength (cols)')
print(tab.round(4).to_string())
print('\ndelta vs no balancing')
print(delta.round(4).to_string())
print('\nshare of simulated test sets where full Sinkhorn beats none:', share_full_beats_none.round(2).to_dict())

keep = [x for x in csv.DictReader(open(LOG, encoding='utf-8')) if not x['id'].startswith('E053')]
with open(LOG, 'w', newline='', encoding='utf-8') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(keep[0].keys()))
    wr.writeheader()
    wr.writerows(keep)
for a in ALPHAS:
    log_row(id=f'E053_alpha={a}', category='postproc', metric='bacc_sim_1000rows',
            description=f'Full Sinkhorn vs none on simulated test sets with Dirichlet({a}) class mix (strong OOF bundle)',
            config=f'N={N}, reps={REPS}, 3 scenarios; strengths {STRENGTHS}', baseline=f'{base[a]:.4f}',
            value=f'{tab.loc[a, 1.0]:.4f}', files='experiments/robust/out/*',
            notes=f"best strength {delta.loc[a].idxmax()} ({delta.loc[a].max():+.4f}); full beats none in "
                  f"{share_full_beats_none[a]:.0%} of sets")
