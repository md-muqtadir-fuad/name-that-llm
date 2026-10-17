"""Confirmatory evaluation of selected grouping configs on fresh seeds (never used for threshold selection).
usage: python confirm.py <tag> <seeds a-b> <regimes e.g. 1-3,1-5,2-4> <configs.json>
configs.json: [{"name":..., "source":..., "algo":..., "thr": float, "cap": 18}, ...]
Per seed and regime: plain / true / recovered balanced accuracy on train rows (splits A and B), pair metrics, and
per-fold (split A) predictions -> out/confirm_<tag>.csv and out/confirm_<tag>_folds.csv."""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc

from common import OUT, PKEY_CODE, SRC, Downstream, pair_metrics, sample_pool
from sources import Pool, run_algo

TAG = sys.argv[1]
S0, S1 = map(int, sys.argv[2].split('-'))
REGIMES = [tuple(map(int, r.split('-'))) for r in sys.argv[3].split(',')]
CFG = json.load(open(sys.argv[4]))
T0 = time.time()
rows_out, fold_rows = [], []
for lo, hi in REGIMES:
    for seed in range(S0, S1 + 1):
        rows = sample_pool(5000 + seed, lo, hi)
        ds = {sp: Downstream(rows, sp) for sp in ('A', 'B')}
        true = PKEY_CODE[rows]
        info = dict(seed=seed, regime=f'{lo}-{hi}', n_pool=len(rows), n_prompts=len(np.unique(true)),
                    n_train=int((SRC[rows] == 'train').sum()))
        res = dict(info, plain_A=ds['A'].plain, plain_B=ds['B'].plain, true_A=ds['A'].evaluate(true)[0],
                   true_B=ds['B'].evaluate(true)[0])
        fo = ds['A'].fo
        yA = ds['A'].y
        preds = {'plain': ds['A'].Lp.argmax(1), 'true': ds['A'].evaluate(true, return_pred=True)}
        pool = Pool(rows, psame=lambda: ds['A'].psame())
        Zc = {}
        for c in CFG:
            S = pool.S(c['source'])
            zc = Zc.setdefault(c['source'], {})
            lab = run_algo(S, c['algo'], c.get('thr', np.nan), zc)
            cap = c.get('cap', 18)
            pm = pair_metrics(rows, lab)
            for sp in ('A', 'B'):
                res[f'{c["name"]}|rec_{sp}'] = ds[sp].evaluate(lab, cap=cap)[0]
            for k in ('P_t', 'R_t', 'F1_t', 'P_x', 'R_x', 'F1_x', 'grp_t', 'pure_t'):
                res[f'{c["name"]}|{k}'] = pm[k]
            preds[c['name']] = ds['A'].evaluate(lab, cap=cap, return_pred=True)
        rows_out.append(res)
        for f in range(5):
            i = fo == f
            fold_rows.append(dict(info, fold=f, n=int(i.sum()),
                                  **{k: float(bacc(yA[i], p[i])) for k, p in preds.items()}))
        print(f'{lo}-{hi} seed {seed} done {time.time() - T0:.0f}s', flush=True)
        pd.DataFrame(rows_out).to_csv(OUT / f'confirm_{TAG}.csv', index=False)
        pd.DataFrame(fold_rows).to_csv(OUT / f'confirm_{TAG}_folds.csv', index=False)
print('done', f'{time.time() - T0:.0f}s')
