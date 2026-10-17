"""Argon suggestions, cheap CPU tests.
(1) Class balancing under SKEWED test priors (fold classes subsampled with Dirichlet(1) weights, 40 draws/fold):
    none | uniform partial Sinkhorn (alpha 0.5, as in v1) | prior-aware: EM prior -> Sinkhorn to it -> divide prior^tau.
    Score set: strong public frozen-LLM OOF blend (calibrated).
(2) Nuisance whitening (PLDA/WCCN-style) vs StandardScaler, and (3) CORAL vs diagonal test-standardisation,
    on the frozen Qwen3-8B L7+22 head (5-fold GroupKFold by prompt)."""
import json
import pathlib
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
tr = pd.read_csv(ROOT / 'data/train.csv')
LABELS = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(LABELS)}).values
g = tr.prompt_id.values
K = 18
folds = list(GroupKFold(5).split(tr, y, g))
T0 = time.time()

# ---------------- (1) balancing under skewed priors
zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
S = sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
S = log_softmax(S * minimize_scalar(lambda t: -log_softmax(S * t, 1)[np.arange(len(y)), y].mean(),
                                    bounds=(0.1, 20), method='bounded').x, 1)
P0 = np.exp(S)


def sinkhorn(P, target, it=50):
    P = P.copy()
    for _ in range(it):
        P *= (target / (P.sum(0) + 1e-12))[None]
        P /= P.sum(1, keepdims=True)
    return P


def uniform_partial(P, a=0.5):
    return sinkhorn(P, (1 - a) * P.sum(0) + a * len(P) / K)


def em_prior(P, eps=0.15, it=30):
    pi = np.full(K, 1 / K)
    for _ in range(it):
        q = pi * P
        q /= q.sum(1, keepdims=True)
        pi = (1 - eps) * q.mean(0) + eps / K
    return pi


def prior_aware(P, tau=0.85):
    pi = em_prior(P)
    Q = sinkhorn(P, len(P) * pi)
    return Q / pi[None] ** tau


rng = np.random.RandomState(0)
res = {k: [] for k in ('none', 'uniform a0.5', 'prior-aware t0.85', 'prior-aware t1.0')}
for _, b in folds:
    for _ in range(40):
        wcls = rng.dirichlet(np.ones(K))
        keep = np.concatenate([rng.choice(b[y[b] == k], min((y[b] == k).sum(), max(2, int(round(len(b[y[b] == k]) * wcls[k] * K / 2)))),
                                          replace=False) if (y[b] == k).sum() >= 2 else b[y[b] == k]
                               for k in range(K)])
        keep = np.unique(keep)
        P = P0[keep]
        res['none'].append(bacc(y[keep], P.argmax(1)))
        res['uniform a0.5'].append(bacc(y[keep], uniform_partial(P).argmax(1)))
        res['prior-aware t0.85'].append(bacc(y[keep], prior_aware(P, 0.85).argmax(1)))
        res['prior-aware t1.0'].append(bacc(y[keep], prior_aware(P, 1.0).argmax(1)))
print('(1) balancing on skewed-prior pools (200 draws), mean bacc and delta vs none:')
for k, v in res.items():
    d = np.array(v) - np.array(res['none'])
    print(f'   {k:20s} {np.mean(v):.4f}  delta {d.mean():+.4f}  (sd {d.std():.4f}, positive {np.mean(d > 0):.2f})')
balanced = [bacc(y[b], uniform_partial(P0[b]).argmax(1)) - bacc(y[b], P0[b].argmax(1)) for _, b in folds]
print(f'   (balanced folds) uniform a0.5 delta {np.mean(balanced):+.4f}', flush=True)

# ---------------- (2)/(3) whitening + CORAL on the Qwen3-8B head
z = np.load(ROOT / 'experiments/papers/qwen3_8b_train_feats.npz')
lays = list(z['layers'])
X = np.concatenate([z['E'][:, lays.index(7)], z['E'][:, lays.index(22)]], 1).astype(np.float64)


def inv_sqrt(C):
    e, V = np.linalg.eigh(C)
    return (V / np.sqrt(np.clip(e, 1e-8, None))) @ V.T


def sqrtm(C):
    e, V = np.linalg.eigh(C)
    return (V * np.sqrt(np.clip(e, 1e-8, None))) @ V.T


def run(name, prep, kdim=512):
    pred = np.zeros(len(y), int)
    for a, b in folds:
        Xa, Xb = prep(a, b, kdim)
        clf = LogisticRegression(C=0.0025, max_iter=3000).fit(Xa, y[a])
        pred[b] = clf.predict(Xb)
    fs = [bacc(y[b], pred[b]) for _, b in folds]
    print(f'{name:52s} {bacc(y, pred):.4f}  folds {np.round(fs, 3)}  ({time.time() - T0:.0f}s)', flush=True)
    return np.array(fs)


def base(a, b, _):
    sc = StandardScaler().fit(X[a])
    return sc.transform(X[a]), (X[b] - X[b].mean(0)) / (X[b].std(0) + 1e-6)


def pca_space(a, b, kdim):
    """Standardise with train stats (test with its own stats), then project to top-k PCs of train."""
    sc = StandardScaler().fit(X[a])
    A = sc.transform(X[a])
    B = (X[b] - X[b].mean(0)) / (X[b].std(0) + 1e-6)
    _, _, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
    V = Vt[:kdim].T
    return A @ V, B @ V


def whiten(alpha, beta):
    def prep(a, b, kdim):
        A, B = pca_space(a, b, kdim)
        Am = pd.DataFrame(A).groupby(g[a]).transform('mean').values           # prompt means per row
        Mm = pd.DataFrame(A).groupby(y[a]).transform('mean').values           # model means per row
        mu = A.mean(0)
        P = pd.DataFrame(A).groupby(g[a]).mean().values - mu
        Sp = P.T @ P / (len(P) - 1)
        E = A - Am - Mm + mu
        Sw = E.T @ E / len(A)
        Sn = (1 - beta) * (alpha * Sp + (1 - alpha) * Sw) + beta * np.diag(np.diag(Sp + Sw))
        Wt = inv_sqrt(Sn)
        A2, B2 = (A - mu) @ Wt, (B - B.mean(0)) @ Wt
        s = A2.std(0) + 1e-9
        return A2 / s, B2 / s
    return prep


def coral(lam):
    def prep(a, b, kdim):
        A, B = pca_space(a, b, kdim)
        Ca = np.cov(A.T) * (1 - lam) + lam * np.eye(A.shape[1])
        Cb = np.cov(B.T) * (1 - lam) + lam * np.eye(B.shape[1])
        Bc = (B - B.mean(0)) @ inv_sqrt(Cb) @ sqrtm(Ca) + A.mean(0)
        return A, Bc
    return prep


print('\n(2)/(3) frozen Qwen3-8B L7+22 head')
b0 = run('base: StandardScaler + test-std (full dims)', base)
b1 = run('PCA-512 space (control for whitening/CORAL)', lambda a, b, k: pca_space(a, b, k))
out = {}
for al, be in ((0.75, 0.1), (0.5, 0.1), (0.75, 0.3)):
    out[f'whiten a{al} b{be}'] = run(f'nuisance whitening alpha {al} beta {be} (PCA-512)', whiten(al, be))
for lam in (0.15, 0.5):
    out[f'CORAL lam{lam}'] = run(f'CORAL lambda {lam} (PCA-512)', coral(lam))
print('\ndelta vs PCA-512 control (mean, positive folds):')
for k, v in out.items():
    d = v - b1
    print(f'   {k:22s} {d.mean():+.4f}  {int((d > 0).sum())}/5')
