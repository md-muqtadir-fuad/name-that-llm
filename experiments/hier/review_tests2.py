"""E042: two more cheap tests from the Argon 4 review (OOF only, CPU).
A) Inductive topic-bias subtraction: Ridge from a topic embedding (content-word TF-IDF -> SVD 256, fitted on the
   training folds) to the prompt-mean OOF logit vector of the training rows; on the held-out fold subtract
   beta * prediction from each passage's logits (no test clustering). Bases: strong (5 folds) and strong+LoRAs (folds 0-2).
B) Sparse-prompt bootstrap for partial Sinkhorn balancing (alpha 0.5, as in the notebooks): full folds vs batches of
   100 sampled prompts keeping 1-4 passages each (p = .4/.3/.2/.1), 300 trials.
Writes experiments/hier/out/review_tests2.csv, paper/experiments.csv rows E042_* (idempotent)."""
import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
from log import log_row  # noqa: E402

OUT = ROOT / 'experiments/hier/out'
tr = pd.read_csv(ROOT / 'data/train.csv')
L = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(L)}).values
P = tr.prompt_id.values
folds = list(GroupKFold(5).split(tr, y, P))
fold = np.zeros(len(tr), int)
for k, (_, b) in enumerate(folds):
    fold[b] = k
pos = {i: k for k, i in enumerate(tr.id.values)}


def cal(Z, m):
    t = minimize_scalar(lambda t: -log_softmax(Z[m] * t, 1)[np.arange(m.sum()), y[m]].mean(), bounds=(0.05, 50), method='bounded').x
    return log_softmax(Z * t, 1)


zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
ALL = np.ones(len(y), bool)
S = cal(sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values()), ALL)


def lora(src):
    Z = np.full((len(tr), 18), np.nan)
    for f in sorted(glob.glob(str(ROOT / src / 'fold*/val_outputs.npz'))):
        z = np.load(f, allow_pickle=True)
        Z[np.array([pos[i] for i in z['ids']])] = z['plain_logits']
    return Z


G, M = lora('nb/lora_gemma_ext/full_out/lora_gemma'), lora('nb/lora_glm_ext/full_out/lora_gemma')
OK3 = ~np.isnan(G[:, 0])
LO = np.full_like(S, np.nan)
LO[OK3] = S[OK3] + 0.5 * cal(np.nan_to_num(G), OK3)[OK3] + 1.0 * cal(np.nan_to_num(M), OK3)[OK3]
res = []


def rec(**kw):
    res.append(kw)
    print(kw, flush=True)


# ---------------------------------------------------------------- A) inductive topic-bias subtraction
for base_name, Z, rows in [('strong', S, ALL), ('strong+loras', LO, OK3)]:
    pred = {b: np.zeros_like(Z) for b in (0.25, 0.5, 1.0)}
    used = sorted(set(fold[rows]))
    for k in used:
        trn, val = rows & (fold != k), rows & (fold == k)
        v = TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=2, ngram_range=(1, 2))
        Xt = v.fit_transform(tr.text[trn])
        svd = TruncatedSVD(256, random_state=0).fit(Xt)
        Et, Ev = svd.transform(Xt), svd.transform(v.transform(tr.text[val]))
        dfz = pd.DataFrame(Z[trn]).assign(p=P[trn])
        pm = dfz.groupby('p').transform('mean').values           # prompt-mean OOF logits of the training rows
        target = pm - Z[trn].mean(0)
        rg = Ridge(alpha=10.0).fit(Et, target)
        hat = rg.predict(Ev)
        for b in pred:
            pred[b][val] = Z[val] - b * hat
    ref = Z[rows].argmax(1)
    for b, Zb in pred.items():
        pb = Zb[rows].argmax(1)
        fd = [bacc(y[rows & (fold == k)], Zb[rows & (fold == k)].argmax(1)) - bacc(y[rows & (fold == k)], Z[rows & (fold == k)].argmax(1)) for k in used]
        rec(test='A_topic_subtract', base=base_name, param=f'beta={b}', bacc=round(bacc(y[rows], pb), 4),
            delta=round(bacc(y[rows], pb) - bacc(y[rows], ref), 4), folds_pos=f'{sum(d > 0 for d in fd)}/{len(fd)}')


# ---------------------------------------------------------------- B) Sinkhorn alpha 0.5: full folds vs sparse batches
def sinkhorn(Z, alpha=0.5):
    Pr = np.exp(Z - Z.max(1, keepdims=True))
    Pr /= Pr.sum(1, keepdims=True)
    target = (1 - alpha) * Pr.sum(0) + alpha * np.full(Pr.shape[1], len(Pr) / Pr.shape[1])
    for _ in range(50):
        Pr *= (target / (Pr.sum(0) + 1e-12))[None, :]
        Pr /= Pr.sum(1, keepdims=True) + 1e-12
    return Pr


rng = np.random.default_rng(0)
for base_name, Z, rows in [('strong', S, ALL), ('strong+loras', LO, OK3)]:
    used = sorted(set(fold[rows]))
    full = [bacc(y[rows & (fold == k)], sinkhorn(Z[rows & (fold == k)]).argmax(1)) - bacc(y[rows & (fold == k)], Z[rows & (fold == k)].argmax(1)) for k in used]
    rec(test='B_sinkhorn', base=base_name, param='full folds (complete prompts)', bacc='', delta=round(float(np.mean(full)), 4),
        folds_pos=f'{sum(d > 0 for d in full)}/{len(full)}')
    prompts = np.unique(P[rows])
    by_p = {p: np.where(rows & (P == p))[0] for p in prompts}
    d = []
    for _ in range(300):
        idx = np.concatenate([rng.choice(by_p[p], min(len(by_p[p]), rng.choice([1, 2, 3, 4], p=[.4, .3, .2, .1])), replace=False)
                              for p in rng.choice(prompts, 100, replace=True)])
        d.append(bacc(y[idx], sinkhorn(Z[idx]).argmax(1)) - bacc(y[idx], Z[idx].argmax(1)))
    d = np.array(d)
    rec(test='B_sinkhorn', base=base_name, param='sparse batches (100 prompts x 1-4 passages, 300 trials)', bacc='',
        delta=round(float(d.mean()), 4), folds_pos=f'{int((d > 0).sum())}/300 trials positive')
R = pd.DataFrame(res)
R.to_csv(OUT / 'review_tests2.csv', index=False)
ex = pd.read_csv(ROOT / 'paper/experiments.csv', dtype=str, keep_default_na=False)
ex[~ex['id'].str.startswith('E042_')].to_csv(ROOT / 'paper/experiments.csv', index=False)
for r in res:
    log_row(id=f"E042_{r['test']}_{r['base']}_{r['param'][:20]}", category='postproc', description=f"review test {r['test']} on {r['base']} ({r['param']})",
            config='experiments/hier/review_tests2.py', metric='oof_bacc', baseline='same base without this step',
            value=str(r['bacc']), delta=str(r['delta']), folds_pos=r['folds_pos'], files='experiments/hier/out/review_tests2.csv')
