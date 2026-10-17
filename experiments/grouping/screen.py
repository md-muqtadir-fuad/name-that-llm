"""Screen grouping candidates on simulated pools (train prompts + external distractor prompts).
usage: python screen.py <tag> <seeds a-b> <lo-hi> <algos comma> [source specs separated by ';'  | 'base']
Thresholds: 10 quantiles of the pool's nearest-neighbour similarity (per source, per pool) -> per-row 'q' column, and
the absolute value in 'thr'. Writes out/screen_<tag>.csv (one row per seed x source x algo x threshold)."""
import sys
import time

import numpy as np
import pandas as pd

from common import OUT, PKEY_CODE, SRC, Downstream, pair_metrics, sample_pool
from sources import Pool, available_base, nn_grid, run_algo

TAG = sys.argv[1]
S0, S1 = map(int, sys.argv[2].split('-'))
LO, HI = map(int, sys.argv[3].split('-'))
ALGOS = sys.argv[4].split(',')
SPECS = available_base() if len(sys.argv) <= 5 or sys.argv[5] == 'base' else sys.argv[5].split(';')
QS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
NOTHR = {'hdbscan', 'average_auto_gmm', 'average_auto_otsu'}
T0 = time.time()
print('specs', SPECS, flush=True)

out = []
for seed in range(S0, S1 + 1):
    rows = sample_pool(1000 + seed, LO, HI)
    ds = {sp: Downstream(rows, sp) for sp in ('A', 'B')}
    true = PKEY_CODE[rows]
    info = dict(seed=seed, regime=f'{LO}-{HI}', n_pool=len(rows), n_prompts=len(np.unique(true)),
                n_train=int((SRC[rows] == 'train').sum()), plain_A=ds['A'].plain, plain_B=ds['B'].plain,
                true_A=ds['A'].evaluate(true)[0], true_B=ds['B'].evaluate(true)[0])
    print(info, f'{time.time() - T0:.0f}s', flush=True)
    pool = Pool(rows, psame=lambda: ds['A'].psame())
    for spec in SPECS:
        t1 = time.time()
        try:
            S = pool.S(spec)
        except Exception as e:
            print('  skip', spec, repr(e), flush=True)
            continue
        grid = nn_grid(S, QS)
        Zc = {}
        for algo in ALGOS:
            for q, thr in (zip(QS, grid) if algo not in NOTHR else [(np.nan, np.nan)]):
                lab = run_algo(S, algo, thr, Zc)
                pm = pair_metrics(rows, lab)
                rA, used = ds['A'].evaluate(lab)
                rB, _ = ds['B'].evaluate(lab)
                out.append(dict(info, source=spec, algo=algo, q=q, thr=thr, rec_A=rA, rec_B=rB, used=used, **pm))
        print(f'  {spec} {time.time() - t1:.0f}s', flush=True)
    pd.DataFrame(out).to_csv(OUT / f'screen_{TAG}.csv', index=False)
print('done', f'{time.time() - T0:.0f}s')
