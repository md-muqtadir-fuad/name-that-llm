"""Does external data help 18-way training (not only gates)? Stylegram 4-view TF-IDF LinearSVC (as v1), 5-fold
GroupKFold by prompt on train; external rows (prvsiyan clean, 18 labels; prompts disjoint from train) are added to
every training fold with sample weight w_ext. Report standalone OOF bacc and the gain when the member is added to the
strong frozen-LLM blend (softmaxed decisions, member weight fitted on a grid; per-fold deltas)."""
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import hstack
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'paper'))
from log import log_row  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
tr = pd.read_csv(ROOT / 'data/train.csv')
L = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(L)}).values
g = tr.prompt_id.values
ext = pd.read_csv(ROOT / 'research/ext_data/prvsiyan/passages_clean.csv')
ext = ext[ext.model.isin(L)].reset_index(drop=True)
ye = ext.model.map({l: i for i, l in enumerate(L)}).values
print('external rows', len(ext), 'per label min/max', np.bincount(ye).min(), np.bincount(ye).max())

FW = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
SPEC = [('char', 3, 5, 250000, 1.0), ('word', 1, 3, 180000, 0.7), ('shape', 2, 6, 60000, 0.5), ('function', 1, 3, 50000, 0.5)]
shape = lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
func = lambda t: " ".join(x.lower() if x.lower() in FW or not x.isalnum() else "CONTENT" for x in re.findall(r"\w+|[^\w\s]", t))


def feats(train_txt, test_txt):
    A, B = [], []
    for an, lo, hi, mf, wt in SPEC:
        tf = shape if an == 'shape' else func if an == 'function' else str
        act = 'char' if an == 'shape' else 'word' if an == 'function' else an
        o = dict(analyzer=act, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf)
        if act == 'word':
            o['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**o)
        A.append(v.fit_transform([tf(t) for t in train_txt]) * wt)
        B.append(v.transform([tf(t) for t in test_txt]) * wt)
    return hstack(A, format='csr'), hstack(B, format='csr')


folds = list(GroupKFold(5).split(tr, y, g))
zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
S = sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
S = log_softmax(S * minimize_scalar(lambda t: -log_softmax(S * t, 1)[np.arange(len(y)), y].mean(),
                                    bounds=(0.1, 20), method='bounded').x, 1)
base = bacc(y, S.argmax(1))
res = {}
for w_ext in (0.0, 0.3, 1.0):
    D = np.zeros((len(y), 18))
    for a, b in folds:
        txt = list(tr.text.values[a]) + (list(ext.text) if w_ext > 0 else [])
        yy = np.concatenate([y[a], ye]) if w_ext > 0 else y[a]
        sw = np.concatenate([np.ones(len(a)), np.full(len(ext), w_ext)]) if w_ext > 0 else None
        A, B = feats(txt, list(tr.text.values[b]))
        D[b] = LinearSVC(C=0.5, max_iter=4000).fit(A, yy, sample_weight=sw).decision_function(B)
    Ls = log_softmax(D * minimize_scalar(lambda t: -log_softmax(D * t, 1)[np.arange(len(y)), y].mean(),
                                         bounds=(0.1, 50), method='bounded').x, 1)
    best = max(((bacc(y, (S + m * Ls).argmax(1)), m) for m in (0.25, 0.5, 0.75, 1.0)))
    fd = [bacc(y[b], (S + best[1] * Ls)[b].argmax(1)) - bacc(y[b], S[b].argmax(1)) for _, b in folds]
    res[w_ext] = (bacc(y, D.argmax(1)), best, fd)
    print(f'w_ext {w_ext}: standalone {res[w_ext][0]:.4f} | strong+{best[1]}*svc {best[0]:.4f} ({best[0] - base:+.4f}) '
          f'folds+ {sum(x > 0 for x in fd)}/5', flush=True)
r0 = res[0.0]
for w_ext in (0.3, 1.0):
    r = res[w_ext]
    log_row(id=f'E025_w{w_ext}', category='data', description=f'18-way stylegram SVC trained on train + prvsiyan external rows (weight {w_ext})',
            config='ext18_test.py', metric='oof_bacc_standalone', baseline=f'{r0[0]:.4f}', value=f'{r[0]:.4f}',
            notes=f'on strong blend: {r[1][0]:.4f} vs no-ext {r0[1][0]:.4f}; folds+ {sum(x > 0 for x in r[2])}/5')
