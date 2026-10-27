"""E064: does NNLS give our extra members weight next to the 37-member OOF bundle? Candidates: Gemma two-stage (E056)
plain and chunk-centred heads (chunks of 2-3 same-prompt passages; the prompt-level head uses oracle 18-way groups and inflated E064 v1 to +0.12) (OOF folds 0-2 only), word-BiLSTM and byte-CNN (E050). Rows = host folds 0-2 (1,620 rows,
where Gemma has OOF); NNLS (log-linear use, as in E062) fit on 2 folds, scored on the 3rd. Compared: bundle only vs
bundle + extras (same rows, same protocol). Reports the extras' mean weight share. CPU, seconds.
Run from the repo root: python experiments/stacking/nnls_gemma.py"""
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import log_softmax
from sklearn.model_selection import GroupKFold

sys.path.insert(0, 'paper')
from log import LOG, log_row  # noqa: E402

OUT = Path('experiments/stacking/out')
train = pd.read_csv('data/train.csv', dtype=str)
labels = np.array(sorted(train.model.unique()))
y = np.searchsorted(labels, train.model.to_numpy())
fold = np.full(len(y), -1)
for k, (_, va) in enumerate(GroupKFold(5).split(np.zeros(len(y)), y, train.prompt_id)):
    fold[va] = k
bacc = lambda yt, p: np.mean([(p[yt == j] == j).mean() for j in np.unique(yt)])
z = np.load('research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)   # our own OOF bundle
pos = pd.Series(np.arange(len(y)), index=train.id).loc[z['ids'].astype(str)].to_numpy()
comp = np.empty_like(z['components'])
comp[:, :, pos] = z['components']                                   # (37, 3, n, 18) in train order
names = list(z['component_names'].astype(str))

extra, enames = [], []
G = {'plain': np.full((len(y), 18), np.nan), 'center': np.full((len(y), 18), np.nan)}
for f in sorted(glob.glob('research/kernels_2026-10-26/gemma_s2/ts/lora_twostage/fold*/val_outputs.npz')):
    v = np.load(f, allow_pickle=True)
    assert list(v['classes']) == list(labels)
    G['plain'][v['rows']] = log_softmax(v['plain_logits'].astype(float), 1)
    G['center'][v['rows']] = log_softmax(v['center_logits_chunk'].astype(float), 1)
for k_, a in G.items():
    extra.append(a)
    enames.append(f'gemma_{k_}')
for m in ('lstm', 'cnn'):
    extra.append(np.load(f'experiments/cnn_lstm/out/oof_{m}.npz')['oof'].astype(float))
    enames.append(m)
rows = np.flatnonzero(np.isfinite(G['plain']).all(1))
assert set(np.unique(fold[rows])) == {0, 1, 2}, np.unique(fold[rows])
Y = np.eye(18)[y]
res = []
for s in range(comp.shape[1]):
    base_lp = comp[:, s]
    for tag, LP in (('bundle', base_lp), ('bundle+extras', np.concatenate([base_lp, np.stack(extra)], 0))):
        P = np.exp(LP - LP.max(2, keepdims=True))
        P /= P.sum(2, keepdims=True)
        pred, W = np.empty(len(y), dtype=int), []
        for k in (0, 1, 2):
            tr = rows[fold[rows] != k]
            te = rows[fold[rows] == k]
            w, _ = nnls(P[:, tr].reshape(len(P), -1).T, Y[tr].reshape(-1))
            W.append(w)
            pred[te] = np.tensordot(w, LP[:, te], 1).argmax(1)
        wm = np.mean(W, 0)
        share = {n: wm[len(names) + i] / wm.sum() for i, n in enumerate(enames)} if tag != 'bundle' else {}
        res.append(dict(scenario=s, setup=tag, bacc=bacc(y[rows], pred[rows]),
                        per_fold=[bacc(y[rows][fold[rows] == k], pred[rows][fold[rows] == k]) for k in (0, 1, 2)], **share))
        print(f"scenario {s} {tag:14s} {res[-1]['bacc']:.4f} " + ' '.join(f'{n} {v:.3f}' for n, v in share.items()), flush=True)
df = pd.DataFrame(res)
df.to_csv(OUT / 'nnls_gemma.csv', index=False)
b = df[df.setup == 'bundle'].reset_index(drop=True)
e = df[df.setup == 'bundle+extras'].reset_index(drop=True)
gain = (e.bacc - b.bacc).mean()
fpos = int(sum((np.array(e.per_fold[i]) > np.array(b.per_fold[i])).sum() for i in range(len(b))))
print(f'mean bundle {b.bacc.mean():.4f} | +extras {e.bacc.mean():.4f} ({gain:+.4f}; fold-scenarios better {fpos}/9) | '
      'mean weight share: ' + ', '.join(f'{n} {e[n].mean():.3f}' for n in enames))

keep = [x for x in csv.DictReader(open(LOG, encoding='utf-8')) if not x['id'].startswith('E064')]
with open(LOG, 'w', newline='', encoding='utf-8') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(keep[0].keys()))
    wr.writeheader()
    wr.writerows(keep)
log_row(id='E064', category='ensemble', metric='oof_bacc_3fold_lofo', description='NNLS weights with extra candidates (Gemma plain/centred, LSTM, CNN) next to the 37-member bundle, host folds 0-2',
        config='scipy nnls (log-linear use), fit on 2 folds, scored on the 3rd; 3 scenarios', baseline=f'{b.bacc.mean():.4f}',
        value=f'{e.bacc.mean():.4f}', folds_pos=f'{fpos}/9', files='experiments/stacking/out/nnls_gemma.csv',
        notes='mean weight share: ' + ', '.join(f'{n} {e[n].mean():.3f}' for n in enames))
