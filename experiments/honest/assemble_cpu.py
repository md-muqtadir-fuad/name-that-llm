"""E065: honest per-row OOF of the CPU members of the public 0.664 stack (tfidf, distort, svc4 on the open pool), from
Kaggle CPU kernel ntl-honest-cpu (0.664 infer.py --sim k, seeded split = RandomState(0) shuffle of sorted prompt ids,
fold = i % 5; members refit per fold, so honest under any split). Checks every train row is scored exactly once,
saves research/honest_oof/cpu_members_oof.npz (train order), reports balanced accuracy per member and of the CPU part
blended with the 0.664 weights (tfidf 0.25, distort 0.10, svc4 12). Run from the repo root."""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bas

sys.path.insert(0, 'paper')
from log import log_row  # noqa: E402

D = 'research/honest_oof/cpu/cache'
train = pd.read_csv('data/train.csv')
labels = sorted(train.model.unique())
y = train.model.map({l: i for i, l in enumerate(labels)}).to_numpy()
rng = np.random.RandomState(0)
pids = np.array(sorted(train.prompt_id.unique()))
rng.shuffle(pids)
fold = train.prompt_id.map({q: i % 5 for i, q in enumerate(pids)}).to_numpy()
pos = pd.Series(np.arange(len(train)), index=train.id.astype(str))
M = {}
seen = np.zeros(len(train), int)
for k in range(5):
    z = np.load(f'{D}/sim{k}_members.npz', allow_pickle=True)   # our own kernel output (ids saved as object array)
    r = pos.loc[z['ids'].astype(str)].to_numpy()
    assert (fold[r] == k).all(), k
    seen[r] += 1
    for m in ('tfidf', 'distort', 'svc4'):
        M.setdefault(m, np.full((len(train), 18), np.nan, np.float32))[r] = z[m]
assert (seen == 1).all(), np.bincount(seen)
np.savez_compressed('research/honest_oof/cpu_members_oof.npz', ids=np.array(train.id.astype(str).tolist()), fold=fold, y=y,
                    labels=np.array(labels), **M)
res = {m: bas(y, v.argmax(1)) for m, v in M.items()}
blend = 0.25 * M['tfidf'] + 0.10 * M['distort'] + 12 * M['svc4']
res['cpu_blend_0664w'] = bas(y, blend.argmax(1))
pf = {m: [round(bas(y[fold == k], v[fold == k].argmax(1)), 3) for k in range(5)] for m, v in M.items()}
for m, v in res.items():
    print(f'{m:16s} {v:.4f} ' + str(pf.get(m, '')))
log_row(id='E065', category='honest-oof', metric='oof_bacc_5fold_seeded', description='honest per-row OOF of the 0.664 CPU members (tfidf, distort, svc4 on the open pool), seeded split',
        config='Kaggle CPU ntl-honest-cpu: 0.664 infer.py --sim k, all 18 passages per held-out prompt, weights 1, members refit per fold',
        baseline='', value=f"{res['svc4']:.4f}", folds_pos='', files='research/honest_oof/cpu_members_oof.npz',
        notes='; '.join(f'{m} {v:.4f}' for m, v in res.items()))
