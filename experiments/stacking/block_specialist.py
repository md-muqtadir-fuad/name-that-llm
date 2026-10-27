"""E063: 3-way specialist for the {claude-fable-5, glm-5.3, glm-5.3-flash} block (H3 recommendation).
Specialist = char 1-5 + word 1-2 TF-IDF + logistic regression trained on pool_v4 rows of the three labels (~15k) plus the
host training-fold rows of those labels; for host rows whose blend top-1 is in the block, the block scores become
blend log-prob + lam * specialist log-prob (lam chosen on the other 4 folds, leave-one-fold-out). Baseline = the strong
OOF bundle reference blend (3 scenarios). Reports overall balanced accuracy and the three recalls. CPU only.
Run from the repo root: python experiments/stacking/block_specialist.py"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, 'paper')
try:   # repo logging helper; absent on Kaggle (results are logged locally from the output CSV)
    from log import LOG, log_row  # noqa: E402
except ImportError:
    LOG = None

BLOCK = ['claude-fable-5', 'glm-5.3', 'glm-5.3-flash']
OUT = Path('experiments/stacking/out')
OUT.mkdir(parents=True, exist_ok=True)
train = pd.read_csv('data/train.csv', dtype=str)
labels = np.array(sorted(train.model.unique()))
y = np.searchsorted(labels, train.model.to_numpy())
bidx = [int(np.flatnonzero(labels == b)[0]) for b in BLOCK]
fold = np.full(len(y), -1)
for k, (_, va) in enumerate(GroupKFold(5).split(np.zeros(len(y)), y, train.prompt_id)):
    fold[va] = k
bacc = lambda yt, p: np.mean([(p[yt == j] == j).mean() for j in np.unique(yt)])
z = np.load('research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)   # our own OOF bundle
pos = pd.Series(np.arange(len(y)), index=train.id).loc[z['ids'].astype(str)].to_numpy()
ref = np.empty_like(z['reference'])
ref[:, pos] = z['reference']
pool = pd.read_parquet('research/ext_data/twostage/pool_v4.parquet', columns=['text', 'model'])
pool = pool[pool.model.isin(BLOCK)].reset_index(drop=True)
print('pool block rows', pool.model.value_counts().to_dict(), flush=True)

spec = np.zeros((len(y), 3))          # specialist log-probs over the block, out of fold for host rows
for k in range(5):
    host_tr = train[(fold != k) & train.model.isin(BLOCK)]
    tx = pd.concat([pool.text, host_tr.text], ignore_index=True)
    ty = np.concatenate([pool.model.map(BLOCK.index).to_numpy(), host_tr.model.map(BLOCK.index).to_numpy()])
    cv = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5), min_df=3, sublinear_tf=True, max_features=300000)
    wv = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=False,
                         token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=200000)
    X = hstack([cv.fit_transform(tx), wv.fit_transform(tx)]).tocsr()
    clf = LogisticRegression(C=2.0, max_iter=3000, class_weight='balanced').fit(X, ty)
    te = np.flatnonzero(fold == k)
    Xte = hstack([cv.transform(train.text.iloc[te]), wv.transform(train.text.iloc[te])]).tocsr()
    spec[te] = clf.predict_log_proba(Xte)
    print(f'fold {k} specialist trained on {X.shape[0]} rows', flush=True)
blk = np.isin(y, bidx)
print('specialist alone, block rows (3-way bacc):', round(bacc(y[blk], np.array(bidx)[spec[blk].argmax(1)]), 4), flush=True)

GRID = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
res = []
for s in range(ref.shape[0]):
    r = ref[s]
    def decide(lam, rows):
        sc = r[rows].copy()
        top_in = np.isin(sc.argmax(1), bidx)
        sub = sc[np.ix_(top_in, bidx)] + lam * spec[rows][top_in]
        out = sc.argmax(1)
        out[top_in] = np.array(bidx)[sub.argmax(1)]
        return out
    pred = np.empty(len(y), dtype=int)
    lams = []
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        lam = max(GRID, key=lambda l: bacc(y[tr], decide(l, tr)))
        lams.append(lam)
        pred[te] = decide(lam, te)
    base = r.argmax(1)
    pf = [bacc(y[fold == f], pred[fold == f]) - bacc(y[fold == f], base[fold == f]) for f in range(5)]
    rec = lambda p, j: (p[y == j] == j).mean()
    res.append(dict(scenario=s, base=bacc(y, base), value=bacc(y, pred), folds_pos=int(np.sum(np.array(pf) > 0)), lams=lams,
                    **{f'rec_{b}_base': rec(base, j) for b, j in zip(BLOCK, bidx)},
                    **{f'rec_{b}_new': rec(pred, j) for b, j in zip(BLOCK, bidx)}))
    print(f"scenario {s}: {res[-1]['base']:.4f} -> {res[-1]['value']:.4f} ({res[-1]['value'] - res[-1]['base']:+.4f}, "
          f"{res[-1]['folds_pos']}/5) lams {lams} | recalls " +
          ', '.join(f"{b} {res[-1][f'rec_{b}_base']:.2f}->{res[-1][f'rec_{b}_new']:.2f}" for b in BLOCK), flush=True)
df = pd.DataFrame(res)
df.to_csv(OUT / 'block_specialist.csv', index=False)
if LOG is None:
    sys.exit(0)

keep =[x for x in csv.DictReader(open(LOG, encoding='utf-8')) if not x['id'].startswith('E063')]
with open(LOG, 'w', newline='', encoding='utf-8') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(keep[0].keys()))
    wr.writeheader()
    wr.writerows(keep)
log_row(id='E063', category='member', metric='oof_bacc_lofo', description='3-way specialist {claude-fable-5, glm-5.3, glm-5.3-flash}: TF-IDF LR on pool_v4 + host folds, re-decides block rows of the blend (lam LOFO)',
        config='char 1-5 + word 1-2 TF-IDF, LR C=2 balanced; lam grid ' + str(GRID), baseline=f'{df.base.mean():.4f}',
        value=f'{df.value.mean():.4f}', folds_pos=f'{df.folds_pos.mean():.1f}/5', files='experiments/stacking/out/block_specialist.csv',
        notes='recalls base->new: ' + ', '.join(f"{b} {df[f'rec_{b}_base'].mean():.2f}->{df[f'rec_{b}_new'].mean():.2f}" for b in BLOCK))
