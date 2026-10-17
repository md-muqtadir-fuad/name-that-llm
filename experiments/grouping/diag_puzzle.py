"""Why did recovered TF-IDF groups (average linkage, q=0.1) beat TRUE groups downstream in screen_test (seed 0, 1-3)?
For seeds a..b: plain / true / rec under dist='ignore' (original harness) and dist='surrogate' (fixed harness), and a
per-train-row decomposition of the rec groups: same train membership as the true group, split of the true group,
merged with other train prompts, contains distractors, oversize (> cap).
usage: python diag_puzzle.py <seeds a-b> [q]"""
import sys

import numpy as np
import pandas as pd

from common import GTR, OUT, PKEY_CODE, SRC, Downstream, sample_pool
from sources import Pool, nn_grid, run_algo

S0, S1 = map(int, sys.argv[1].split('-'))
Q = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
res, cats = [], []
for seed in range(S0, S1 + 1):
    rows = sample_pool(1000 + seed, 1, 3)
    true = PKEY_CODE[rows]
    pool = Pool(rows)
    S = pool.S('tfidf')
    thr = nn_grid(S, (Q,))[0]
    lab = run_algo(S, 'average', thr)
    r = dict(seed=seed, thr=thr)
    for sp in ('A', 'B'):
        for dist in ('ignore', 'surrogate'):
            ds = Downstream(rows, sp, dist=dist)
            r[f'plain_{sp}'] = ds.plain
            r[f'true_{sp}_{dist}'] = ds.evaluate(true)[0]
            r[f'rec_{sp}_{dist}'], r[f'used_{sp}_{dist}'] = ds.evaluate(lab)
            if sp == 'A':
                pt = ds.evaluate(true, return_pred=True)
                pr = ds.evaluate(lab, return_pred=True)
                pp = ds.Lp.argmax(1)
                y = ds.y
                tp = ds.tr_pos
                # categories per train row
                L = pd.factorize(lab)[0]
                sz = np.bincount(L)[L]
                is_tr = SRC[rows] == 'train'
                df = pd.DataFrame({'g': L, 'p': true, 'tr': is_tr})
                ntr_in_g = df[df.tr].groupby('g').size()
                nprm_in_g = df[df.tr].groupby('g')['p'].nunique()
                ndis_in_g = df[~df.tr].groupby('g').size()
                tsz = df[df.tr].groupby('p').size()
                same_in_g = df[df.tr].groupby(['g', 'p']).size()
                cat = []
                for i in tp:
                    g, p = L[i], true[i]
                    nd = int(ndis_in_g.get(g, 0))
                    if sz[i] > 18:
                        c = 'oversize'
                    elif sz[i] == 1:
                        c = 'singleton'
                    elif nprm_in_g[g] > 1:
                        c = 'merged_train'
                    elif same_in_g[(g, p)] == 1 and nd > 0:
                        c = 'alone_with_distractors'
                    elif same_in_g[(g, p)] < tsz[p]:
                        c = 'split' + ('+dist' if nd else '')
                    else:
                        c = 'exact' + ('+dist' if nd else '')
                    cat.append(c)
                cats.append(pd.DataFrame({'seed': seed, 'dist': dist, 'cat': cat, 'y': y, 'plain': pp == y,
                                          'true': pt == y, 'rec': pr == y, 'tsz': tsz[true[tp]].values}))
    res.append(r)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
R = pd.DataFrame(res)
R.to_csv(OUT / f'diag_puzzle_q{Q}.csv', index=False)
for dist in ('ignore', 'surrogate'):
    g_true = ((R[f'true_A_{dist}'] - R.plain_A) + (R[f'true_B_{dist}'] - R.plain_B)) / 2
    g_rec = ((R[f'rec_A_{dist}'] - R.plain_A) + (R[f'rec_B_{dist}'] - R.plain_B)) / 2
    d = g_rec - g_true
    print(f'{dist:9s}: true gain {g_true.mean():+.4f} sd {g_true.std():.4f} | rec gain {g_rec.mean():+.4f} '
          f'sd {g_rec.std():.4f} | rec-true {d.mean():+.4f} se {d.std() / np.sqrt(len(d)):.4f} (n={len(d)})')
C = pd.concat(cats)
C.to_csv(OUT / f'diag_puzzle_cats_q{Q}.csv', index=False)
t = C.groupby(['dist', 'cat']).agg(n=('y', 'size'), plain=('plain', 'mean'), true=('true', 'mean'),
                                   rec=('rec', 'mean')).reset_index()
t['rec-true'] = t['rec'] - t['true']
print(t.round(4).to_string(index=False))
