"""E054: uncertainty-routed member weighting. Adds a member's OOF log-probs to the strong OOF blend
(research/ext_oof/q35heads reference, 3 scenarios) either with one fixed weight, or with a separate weight per confidence
band of the blend (terciles of the top-1 minus top-2 probability margin; band edges fitted on the training folds).
All weights are chosen on the other 4 prompt-grouped folds and applied to the held-out fold (leave-one-fold-out).
Members: E050 word-BiLSTM and byte-CNN OOF. CPU only. Run from the repo root: python experiments/uncertainty/routed_weight.py"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, 'paper')
from log import LOG, log_row  # noqa: E402

OUT = Path('experiments/uncertainty/out')
OUT.mkdir(parents=True, exist_ok=True)
train = pd.read_csv('data/train.csv', dtype=str)
labels = np.array(sorted(train.model.unique()))
y = np.searchsorted(labels, train.model.to_numpy())
fold = np.full(len(y), -1)
for k, (_, va) in enumerate(GroupKFold(5).split(np.zeros(len(y)), y, train.prompt_id)):
    fold[va] = k
bacc = lambda yt, p: np.mean([(p[yt == j] == j).mean() for j in np.unique(yt)])
z = np.load('research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)   # our own OOF bundle
pos = pd.Series(np.arange(len(y)), index=train.id).loc[z['ids'].astype(str)].to_numpy()
ref = np.empty_like(z['reference'])
ref[:, pos] = z['reference']
members = {m: np.load(f'experiments/cnn_lstm/out/oof_{m}.npz')['oof'].astype(float) for m in ('lstm', 'cnn')}
GRID = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
NB = 3


def margin_of(lp):
    p = np.exp(lp - lp.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def fit_fixed(r, m, rows):
    return max(GRID, key=lambda w: bacc(y[rows], (r[rows] + w * m[rows]).argmax(1)))


def fit_banded(r, m, rows, band):
    """coordinate ascent over per-band weights, starting from the best fixed weight; objective = bacc on `rows`."""
    w = np.full(NB, fit_fixed(r, m, rows))
    for _ in range(2):
        for b in range(NB):
            def score(v):
                ww = w.copy()
                ww[b] = v
                return bacc(y[rows], (r[rows] + ww[band[rows]][:, None] * m[rows]).argmax(1))
            w[b] = max(GRID, key=score)
    return w


rows_out = []
for name, m in members.items():
    for s in range(ref.shape[0]):
        r = ref[s]
        mg = margin_of(r)
        pred_fixed = np.empty(len(y), dtype=int)
        pred_band = np.empty(len(y), dtype=int)
        for k in range(5):
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            edges = np.quantile(mg[tr], [1 / 3, 2 / 3])          # band edges from training folds only
            band = np.digitize(mg, edges)                         # 0 = most uncertain third
            wf = fit_fixed(r, m, tr)
            wb = fit_banded(r, m, tr, band)
            pred_fixed[te] = (r[te] + wf * m[te]).argmax(1)
            pred_band[te] = (r[te] + wb[band[te]][:, None] * m[te]).argmax(1)
            rows_out.append(dict(member=name, scenario=s, fold=k, w_fixed=wf, w_uncertain=wb[0], w_middle=wb[1],
                                 w_confident=wb[2], base=bacc(y[te], r[te].argmax(1)),
                                 fixed=bacc(y[te], pred_fixed[te]), banded=bacc(y[te], pred_band[te])))
        rows_out.append(dict(member=name, scenario=s, fold='all', base=bacc(y, r.argmax(1)),
                             fixed=bacc(y, pred_fixed), banded=bacc(y, pred_band)))
df = pd.DataFrame(rows_out)
df.to_csv(OUT / 'routed_weight.csv', index=False)

keep = [x for x in csv.DictReader(open(LOG, encoding='utf-8')) if not x['id'].startswith('E054')]
with open(LOG, 'w', newline='', encoding='utf-8') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(keep[0].keys()))
    wr.writeheader()
    wr.writerows(keep)
for name in members:
    d = df[df.member == name]
    al = d[d.fold == 'all']
    pf = d[d.fold != 'all'].groupby('fold')[['fixed', 'banded']].mean()
    fpos = int((pf.banded - pf.fixed).gt(0).sum())
    wts = d[d.fold != 'all'][['w_fixed', 'w_uncertain', 'w_middle', 'w_confident']].mean()
    print(f"{name}: base {al.base.mean():.4f} | fixed weight {al.fixed.mean():.4f} | confidence-banded "
          f"{al.banded.mean():.4f} ({al.banded.mean() - al.fixed.mean():+.4f} vs fixed, {fpos}/5 folds) | mean weights "
          + ', '.join(f'{k[2:]} {v:.2f}' for k, v in wts.items()))
    log_row(id=f'E054_{name}', category='postproc', metric='oof_bacc_lofo',
            description=f'Uncertainty-routed weight for {name} (3 confidence bands of the blend) vs one fixed weight, '
                        'added to the strong OOF blend; weights chosen leave-one-fold-out',
            config=f'grid {GRID}; bands = terciles of top1-top2 margin (train folds)', baseline=f'{al.fixed.mean():.4f}',
            value=f'{al.banded.mean():.4f}', folds_pos=f'{fpos}/5', files='experiments/uncertainty/out/routed_weight.csv',
            notes='mean weights ' + ', '.join(f'{k[2:]} {v:.2f}' for k, v in wts.items()))
