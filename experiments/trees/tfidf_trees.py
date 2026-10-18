"""Tree models instead of LinearSVC on the 4-view stylegram TF-IDF (prompt-grouped 5-fold OOF on train).
Variants (all fitted inside each fold):
  svc        LinearSVC C=0.5 (reference, as experiments/speechmap/eval.py)
  lgb_chi2   LightGBM on chi2 top-20k TF-IDF columns
  et_chi2    ExtraTrees(500) on the same columns
  lgb_svd    LightGBM on TruncatedSVD(256) of the TF-IDF
Reports standalone OOF bacc and the gain when added to the strong frozen-LLM OOF blend (member weight grid).
Writes experiments/trees/out/results.csv and appends paper/experiments.csv."""
import json
import pathlib
import re
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import hstack
from scipy.special import log_softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
from log import log_row  # noqa: E402

OUT = ROOT / 'experiments/trees/out'
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
tr = pd.read_csv(ROOT / 'data/train.csv')
L = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(L)}).values
folds = list(GroupKFold(5).split(tr, y, tr.prompt_id.values))

zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
S = sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
cal = lambda Z: log_softmax(Z * minimize_scalar(lambda t: -log_softmax(Z * t, 1)[np.arange(len(y)), y].mean(),
                                                bounds=(0.05, 50), method='bounded').x, 1)
S = cal(S)
base = bacc(y, S.argmax(1))

FW = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
SPEC = [('char', 3, 5, 250000, 1.0), ('word', 1, 3, 180000, 0.7), ('shape', 2, 6, 60000, 0.5), ('function', 1, 3, 50000, 0.5)]
shape = lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
func = lambda t: " ".join(x.lower() if x.lower() in FW or not x.isalnum() else "CONTENT" for x in re.findall(r"\w+|[^\w\s]", t))


def tfidf(txt, test_txt):
    A, B = [], []
    for an, lo, hi, mf, wt in SPEC:
        tf = shape if an == 'shape' else func if an == 'function' else str
        act = 'char' if an == 'shape' else 'word' if an == 'function' else an
        o = dict(analyzer=act, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf)
        if act == 'word':
            o['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**o)
        A.append(v.fit_transform([tf(t) for t in txt]) * wt)
        B.append(v.transform([tf(t) for t in test_txt]) * wt)
    return hstack(A, format='csr'), hstack(B, format='csr')


LGB = dict(objective='multiclass', num_class=18, learning_rate=0.05, num_leaves=15, min_data_in_leaf=5,
           feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=0)
VARIANTS = ['svc', 'lgb_chi2', 'et_chi2', 'lgb_svd']
OOF = {k: np.zeros((len(y), 18)) for k in VARIANTS}
for fi, (a, b) in enumerate(folds):
    Xa, Xb = tfidf(list(tr.text.values[a]), list(tr.text.values[b]))
    m = LinearSVC(C=0.5, max_iter=4000).fit(Xa, y[a])
    OOF['svc'][b] = m.decision_function(Xb)
    sel = SelectKBest(chi2, k=20000).fit(Xa, y[a])
    Ca, Cb = sel.transform(Xa).astype(np.float32), sel.transform(Xb).astype(np.float32)
    OOF['lgb_chi2'][b] = np.log(np.clip(lgb.train(LGB, lgb.Dataset(Ca, y[a]), 400).predict(Cb), 1e-9, 1))
    et = ExtraTreesClassifier(500, max_features='sqrt', min_samples_leaf=1, n_jobs=4, random_state=0).fit(Ca, y[a])
    OOF['et_chi2'][b] = np.log(np.clip(et.predict_proba(Cb), 1e-4, 1))
    svd = TruncatedSVD(256, random_state=0).fit(Xa)
    Da, Db = svd.transform(Xa), svd.transform(Xb)
    OOF['lgb_svd'][b] = np.log(np.clip(lgb.train(LGB, lgb.Dataset(Da, y[a]), 400).predict(Db), 1e-9, 1))
    print(f'fold {fi} done {round(time.time() - T0)}s', {k: round(bacc(y[b], OOF[k][b].argmax(1)), 4) for k in VARIANTS},
          flush=True)

res = []
for k in VARIANTS:
    Lk = cal(OOF[k])
    best = max((bacc(y, (S + mw * Lk).argmax(1)), mw) for mw in (0.1, 0.25, 0.5, 0.75, 1.0))
    fd = [bacc(y[b], (S + best[1] * Lk)[b].argmax(1)) - bacc(y[b], S[b].argmax(1)) for _, b in folds]
    res.append(dict(variant=k, standalone=round(bacc(y, OOF[k].argmax(1)), 4), on_strong=round(best[0], 4),
                    delta_vs_strong=round(best[0] - base, 4), member_w=best[1], folds_pos=f'{sum(x > 0 for x in fd)}/5'))
# trees as an extra member on top of svc (diversity check)
Ls = cal(OOF['svc'])
for k in ('lgb_chi2', 'et_chi2', 'lgb_svd'):
    Lk = cal(OOF[k])
    best = max((bacc(y, (S + 0.25 * Ls + mw * Lk).argmax(1)), mw) for mw in (0.1, 0.25, 0.5))
    ref = bacc(y, (S + 0.25 * Ls).argmax(1))
    res.append(dict(variant=f'svc+{k}', standalone='', on_strong=round(best[0], 4), delta_vs_strong=round(best[0] - ref, 4),
                    member_w=best[1], folds_pos='', notes='delta vs strong+svc'))
R = pd.DataFrame(res)
R.to_csv(OUT / 'results.csv', index=False)
print(R.to_string(), f'\nbase {base:.4f}  {round(time.time() - T0)}s')
for r in res:
    log_row(id=f"E032_{r['variant']}", category='model', description=f"tree model vs SVC on 4-view TF-IDF ({r['variant']})",
            config='experiments/trees/tfidf_trees.py', metric='oof_bacc_on_strong', baseline=f'{base:.4f}',
            value=str(r['on_strong']), delta=str(r['delta_vs_strong']), folds_pos=r['folds_pos'],
            files='experiments/trees/out/results.csv', notes=f"standalone {r['standalone']}; {r.get('notes', '')}")
