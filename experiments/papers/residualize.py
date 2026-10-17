"""Group-free topic removal. Singletons dominate the test, so the topic offset must be estimated from the passage itself.
B: predicted-topic residualisation. Kernel ridge (cosine kernel on content-word TF-IDF) learns c_i -> mean feature of the
   passage's prompt (averaged over all ~18 models, so it carries no model style). x~ = x - lam * f(c).
   Cross-fitted: train rows get inner-fold predictions; held-out rows get predictions from the training fold.
H: length residualisation with a within-model slope: x~ = x - beta * (loglen - mean).
Base = frozen Qwen3-8B L7+22 head (StandardScaler + LR C=0.0025, test-standardised), 5-fold GroupKFold by prompt."""
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, normalize

ROOT = pathlib.Path(__file__).resolve().parents[2]
tr = pd.read_csv(ROOT / 'data/train.csv')
y = tr.model.astype('category').cat.codes.values
g = tr.prompt_id.values
z = np.load(ROOT / 'experiments/papers/qwen3_8b_train_feats.npz')
lays = list(z['layers'])
X = np.concatenate([z['E'][:, lays.index(7)], z['E'][:, lays.index(22)]], 1).astype(np.float64)
ll = np.log(tr.text.str.split().str.len().values.astype(float))
C = normalize(TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=2).fit_transform(tr.text))  # unsupervised
folds = list(GroupKFold(5).split(X, y, g))
T0 = time.time()


def prompt_means(rows):
    df = pd.DataFrame(X[rows])
    return df.groupby(g[rows]).transform('mean').values


def topic_fit_predict(fit_rows, pred_rows, alpha, Cmat):
    M = prompt_means(fit_rows)
    mu = M.mean(0)
    K = (Cmat[fit_rows] @ Cmat[fit_rows].T)
    K = K.toarray() if hasattr(K, 'toarray') else K
    A = np.linalg.solve(K + alpha * np.eye(len(fit_rows)), M - mu)
    Kp = Cmat[pred_rows] @ Cmat[fit_rows].T
    Kp = Kp.toarray() if hasattr(Kp, 'toarray') else Kp
    return Kp @ A                      # predicted deviation of the prompt mean from the global mean


def residual_topic(a, b, alpha, lam, Cmat):
    Ra = np.zeros((len(a), X.shape[1]))
    for ia, ib in GroupKFold(4).split(a, groups=g[a]):   # inner cross-fit for training rows
        Ra[ib] = topic_fit_predict(a[ia], a[ib], alpha, Cmat)
    Rb = topic_fit_predict(a, b, alpha, Cmat)
    return X[a] - lam * Ra, X[b] - lam * Rb


def residual_len(Xa, Xb, a, b):
    ya, la = y[a], ll[a]
    Xd = Xa - pd.DataFrame(Xa).groupby(ya).transform('mean').values
    ld = la - pd.Series(la).groupby(ya).transform('mean').values
    beta = (ld @ Xd) / (ld @ ld)
    m = la.mean()
    return Xa - np.outer(la - m, beta), Xb - np.outer(ll[b] - m, beta)


def run(name, transform):
    P = np.zeros(len(y), int)
    for a, b in folds:
        Xa, Xb = transform(a, b)
        sc = StandardScaler().fit(Xa)
        clf = LogisticRegression(C=0.0025, max_iter=3000).fit(sc.transform(Xa), y[a])
        P[b] = clf.predict((Xb - Xb.mean(0)) / (Xb.std(0) + 1e-6))
    fold_s = [bacc(y[b], P[b]) for _, b in folds]
    print(f'{name:45s} {bacc(y, P):.4f}  folds {np.round(fold_s, 3)}  ({time.time() - T0:.0f}s)', flush=True)
    return fold_s


base = run('base', lambda a, b: (X[a], X[b]))
res = {}
res['H length'] = run('H: length residual', lambda a, b: residual_len(X[a], X[b], a, b))
for alpha in (0.3, 1.0):
    for lam in (0.5, 1.0):
        res[f'B a{alpha} l{lam}'] = run(f'B: topic residual (tfidf) alpha {alpha} lam {lam}',
                                        lambda a, b, al=alpha, lm=lam: residual_topic(a, b, al, lm, C))
Xn = normalize(X - X.mean(0))
res['B llm'] = run('B: topic residual (centred LLM-feature kernel) a1 l1',
                   lambda a, b: residual_topic(a, b, 1.0, 1.0, Xn))
best = max((k for k in res if k.startswith('B a')), key=lambda k: np.mean(res[k]))
al, lm = float(best.split()[1][1:]), float(best.split()[2][1:])
res['B+H'] = run(f'B({best})+H', lambda a, b: residual_len(*residual_topic(a, b, al, lm, C), a, b))
print('\nfold-wise delta vs base (positive folds / 5):')
for k, v in res.items():
    d = np.array(v) - np.array(base)
    print(f'  {k:12s} mean {d.mean():+.4f}  pos {int((d > 0).sum())}/5')
