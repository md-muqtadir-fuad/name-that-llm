"""CORAL under a realistic test pool. Covariance/mean for test alignment come from a POOL = held-out rows sampled
1-3 per prompt + ~3500 distractor passages from other prompts (Oleg corpus, 1 per prompt). Evaluation on all held-out
rows of the fold (they are transformed with the pool statistics). Compared with the optimistic setup (pool = whole
held-out fold, 30 complete prompts) used in argon_tests.py."""
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
tr = pd.read_csv(ROOT / 'data/train.csv')
y = tr.model.astype('category').cat.codes.values
g = tr.prompt_id.values
z = np.load(ROOT / 'experiments/papers/qwen3_8b_train_feats.npz')
lays = list(z['layers'])
X = np.concatenate([z['E'][:, lays.index(7)], z['E'][:, lays.index(22)]], 1).astype(np.float64)
import sys
d = np.load(ROOT / 'experiments/papers' / (sys.argv[1] if len(sys.argv) > 1 else 'qwen3_8b_distractor_feats.npz'))
D = np.concatenate([d['E'][:, 0], d['E'][:, 1]], 1).astype(np.float64)   # layers [7, 22]
print('train', X.shape, 'distractors', D.shape)
folds = list(GroupKFold(5).split(X, y, g))
rng = np.random.RandomState(0)
T0 = time.time()


def msqrt(C, p):
    e, V = np.linalg.eigh(C)
    return (V * np.clip(e, 1e-8, None) ** p) @ V.T


res = {}
for fi, (a, b) in enumerate(folds):
    sc = StandardScaler().fit(X[a])
    A = sc.transform(X[a])
    _, _, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
    V = Vt[:512].T
    Ap = A @ V
    clf = LogisticRegression(C=0.0025, max_iter=3000).fit(Ap, y[a])
    clf_full = LogisticRegression(C=0.0025, max_iter=3000).fit(A, y[a])
    Ca = np.cov(Ap.T)
    for setting in ('optimistic', 'realistic', 'realistic_nodis'):
        for draw in range(3 if setting != 'optimistic' else 1):
            if setting == 'optimistic':
                pool = X[b]
            else:
                sub = np.concatenate([rng.choice(b[g[b] == p], rng.randint(1, 4), replace=False) for p in np.unique(g[b])])
                pool = np.vstack([X[sub], D]) if setting == 'realistic' else X[sub]
            mu, sd = pool.mean(0), pool.std(0) + 1e-6
            Bf = (X[b] - mu) / sd                          # test-std with pool statistics (as in the notebook)
            Pf = (pool - mu) / sd
            Bp, Pp = Bf @ V, Pf @ V
            out = {'full test-std': clf_full.predict(Bf), 'PCA512 test-std': clf.predict(Bp)}
            Cp = np.cov(Pp.T)
            for lam in (0.15, 0.5, 0.8):
                Cb = (1 - lam) * Cp + lam * np.eye(len(Cp))
                Ca2 = (1 - lam) * Ca + lam * np.eye(len(Ca))
                T = msqrt(Cb, -0.5) @ msqrt(Ca2, 0.5)
                out[f'CORAL lam{lam}'] = clf.predict((Bp - Pp.mean(0)) @ T + Ap.mean(0))
            for k, p in out.items():
                res.setdefault((setting, k), []).append(bacc(y[b], p))
    print(f'fold {fi} done {time.time() - T0:.0f}s', flush=True)
print()
for (s, k), v in sorted(res.items()):
    base = np.mean(res[(s, 'full test-std')])
    print(f'{s:16s} {k:18s} {np.mean(v):.4f}  delta vs full test-std {np.mean(v) - base:+.4f}')
