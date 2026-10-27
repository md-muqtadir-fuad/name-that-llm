"""E062: NNLS stacking weights vs hand-set weights (test plan #15). On our strong OOF bundle (37 member OOF
log-probs, research/ext_oof/q35heads, 3 scenarios), fit non-negative member weights w by NNLS:
    min_w || sum_m w_m * P_m - Y ||^2,  w >= 0   (P_m = member probabilities, Y = one-hot labels, all rows x classes)
on 4 prompt-grouped folds and score balanced accuracy on the 5th (leave-one-fold-out). Two blend spaces: probabilities
(NNLS weights used as a probability mixture) and log-probabilities (same weights in a log-linear blend). Baselines:
the bundle's reference blend (manifest weights) and equal weights over the same 37 members. CPU only.
Run from the repo root: python experiments/stacking/nnls_blend.py"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold

sys.path.insert(0, 'paper')
from log import LOG, log_row  # noqa: E402

OUT = Path('experiments/stacking/out')
OUT.mkdir(parents=True, exist_ok=True)
z = np.load('research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)   # our own OOF bundle
classes = z['classes'].astype(str)
train = pd.read_csv('data/train.csv', dtype=str).set_index('id').loc[z['ids'].astype(str)]
y = np.searchsorted(classes, train.model.to_numpy())
fold = np.full(len(y), -1)
for k, (_, va) in enumerate(GroupKFold(5).split(np.zeros(len(y)), y, train.prompt_id)):
    fold[va] = k
bacc = lambda yt, p: np.mean([(p[yt == j] == j).mean() for j in np.unique(yt)])
names = z['component_names'].astype(str)
Y = np.eye(18)[y]
rows = []
for s in range(z['components'].shape[1]):
    LP = z['components'][:, s]                                    # (37, n, 18) log-probs
    P = np.exp(LP - LP.max(2, keepdims=True))
    P /= P.sum(2, keepdims=True)
    preds = {k: np.empty(len(y), dtype=int) for k in ('nnls_prob', 'nnls_logp', 'equal_logp')}
    W_all = []
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        A = P[:, tr].reshape(len(names), -1).T                      # (n_tr*18, 37)
        w, _ = nnls(A, Y[tr].reshape(-1))
        W_all.append(w)
        preds['nnls_prob'][te] = np.tensordot(w, P[:, te], 1).argmax(1)
        preds['nnls_logp'][te] = np.tensordot(w, LP[:, te], 1).argmax(1)
        preds['equal_logp'][te] = LP[:, te].sum(0).argmax(1)
    ref = z['reference'][s].argmax(1)
    for k, p in preds.items():
        pf = [bacc(y[fold == f], p[fold == f]) - bacc(y[fold == f], ref[fold == f]) for f in range(5)]
        rows.append(dict(scenario=s, method=k, bacc=bacc(y, p), reference=bacc(y, ref), folds_pos=int(np.sum(np.array(pf) > 0))))
    wm = np.mean(W_all, 0)
    top = np.argsort(-wm)[:8]
    print(f'scenario {s}: reference {bacc(y, ref):.4f} | ' + ' | '.join(f"{r['method']} {r['bacc']:.4f}" for r in rows[-3:])
          + f" | nonzero members {int((wm > 1e-6).sum())}/37; top: " + ', '.join(f'{names[i]} {wm[i]:.2f}' for i in top))
df = pd.DataFrame(rows)
df.to_csv(OUT / 'nnls_blend.csv', index=False)
summ = df.groupby('method')[['bacc', 'reference', 'folds_pos']].mean()
print(summ.round(4).to_string())

keep = [x for x in csv.DictReader(open(LOG, encoding='utf-8')) if not x['id'].startswith('E062')]
with open(LOG, 'w', newline='', encoding='utf-8') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(keep[0].keys()))
    wr.writeheader()
    wr.writerows(keep)
for m, r in summ.iterrows():
    log_row(id=f'E062_{m}', category='ensemble', metric='oof_bacc_lofo', description=f'NNLS stacking ({m}) vs hand-set reference weights on the 37-member OOF bundle',
            config='scipy nnls on one-hot targets, weights fit on 4 folds, applied to the 5th; mean of 3 scenarios',
            baseline=f'{r.reference:.4f}', value=f'{r.bacc:.4f}', folds_pos=f'{r.folds_pos:.1f}/5',
            files='experiments/stacking/out/nnls_blend.csv')
