"""OOF blend accuracy per passage-length bucket (train, 3 scenarios), overall and for the sibling classes."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score

OUT = Path('research/bottleneck/h2_surface/out'); OUT.mkdir(parents=True, exist_ok=True)
train = pd.read_csv('data/train.csv')
z = np.load('research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)
CL = list(z['classes']); REF = z['reference']
assert (z['ids'] == train.id.values).all()
y = train.model.map({c: i for i, c in enumerate(CL)}).to_numpy()
nw = train.text.str.split().str.len().to_numpy()
bins = [0, 70, 110, 160, 10 ** 6]; names = ['<70', '70-110', '110-160', '>160']
bucket = pd.cut(nw, bins, right=False, labels=names).astype(str)
SIB = {'gemini36/37/38': ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'], 'glm53/flash': ['glm-5.3', 'glm-5.3-flash'],
       'gpt56 luna/sol/terra': ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'], 'opus48/sonnet5': ['claude-opus-4-8', 'claude-sonnet-5']}
sib_of = np.full(18, -1)
for k, v in enumerate(SIB.values()):
    for c in v:
        sib_of[CL.index(c)] = k
is_sib = sib_of[y] >= 0
rows = []
for s, sname in enumerate(['s1', 's3', 's6']):
    pr = REF[s].argmax(1)
    for b in names + ['all']:
        m = np.ones(len(y), bool) if b == 'all' else (bucket == b)
        ms = m & is_sib
        r = dict(scenario=sname, bucket=b, n=int(m.sum()), acc=(pr[m] == y[m]).mean(), bacc=balanced_accuracy_score(y[m], pr[m]),
                 n_sib=int(ms.sum()), sib_acc=(pr[ms] == y[ms]).mean(),
                 sib_confusion=((sib_of[pr[ms]] == sib_of[y[ms]]) & (pr[ms] != y[ms])).mean(),
                 nonsib_acc=(pr[m & ~is_sib] == y[m & ~is_sib]).mean())
        for k, (g, v) in enumerate(SIB.items()):
            mg = m & (sib_of[y] == k)
            r[f'acc[{g}]'] = (pr[mg] == y[mg]).mean() if mg.any() else np.nan
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(OUT / 'length_bucket_accuracy.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(res.round(3).to_string())
# length distribution by sibling model (train)
L = pd.DataFrame({'model': train.model, 'nw': nw, 'bucket': bucket})
d = L[L.model.isin(sum(SIB.values(), []))].groupby('model').nw.describe()[['mean', '25%', '50%', '75%']]
print(d.round(1).to_string())
print(pd.crosstab(L.model, L.bucket)[names].to_string())
d.to_csv(OUT / 'sibling_length_by_model.csv')
