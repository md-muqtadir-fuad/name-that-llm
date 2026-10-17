"""Transductive nuisance whitening (Gemini/Argon #1). Per fold, in the PCA-512 space of the Qwen3-8B L7+22 features:
Sigma_B = cov(train prompt means), Sigma_W = cov(prompt-centred train rows), Sigma_T = cov(held-out rows, unlabeled),
Sigma_nuis = (Sigma_T - Sigma_W)_+ ; Phi = (l1 I + l2 Sigma_B + l3 Sigma_nuis + l4 Sigma_W)^(-1/2), each Sigma
normalised by trace/d. Train LR on Phi * (prompt-centred train) [or uncentred], predict on Phi * (test - test mean)."""
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
folds = list(GroupKFold(5).split(X, y, g))
T0 = time.time()
prep = []
for a, b in folds:
    sc = StandardScaler().fit(X[a])
    A, B = sc.transform(X[a]), (X[b] - X[b].mean(0)) / (X[b].std(0) + 1e-6)
    _, _, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
    A, B = A @ Vt[:512].T, B @ Vt[:512].T
    Am = pd.DataFrame(A).groupby(g[a]).transform('mean').values
    R = A - Am
    Pm = pd.DataFrame(A).groupby(g[a]).mean().values
    nrm = lambda S: S / (np.trace(S) / len(S))
    SB = nrm(np.cov((Pm - Pm.mean(0)).T))
    SW = nrm(np.cov(R.T))
    ST = np.cov((B - B.mean(0)).T)
    D = ST - np.cov(R.T) * (np.trace(ST) / np.trace(ST))   # same scale as raw covariances
    e, V = np.linalg.eigh(ST - np.cov(R.T))
    SN = nrm((V * np.clip(e, 0, None)) @ V.T + 1e-9 * np.eye(len(e)))
    prep.append((A, B, R, SB, SW, SN))
print(f'prep {time.time() - T0:.0f}s', flush=True)


def run(name, l, C, centred):
    pred = np.zeros(len(y), int)
    for (a, b), (A, B, R, SB, SW, SN) in zip(folds, prep):
        M = l[0] * np.eye(A.shape[1]) + l[1] * SB + l[2] * SN + l[3] * SW
        e, V = np.linalg.eigh(M)
        Phi = (V / np.sqrt(e)) @ V.T
        Tr = (R if centred else A - A.mean(0)) @ Phi
        s = Tr.std(0) + 1e-9
        clf = LogisticRegression(C=C, max_iter=3000).fit(Tr / s, y[a])
        pred[b] = clf.predict(((B - B.mean(0)) @ Phi) / s)
    fs = [bacc(y[b], pred[b]) for _, b in folds]
    print(f'{name:58s} {bacc(y, pred):.4f}  folds {np.round(fs, 3)}', flush=True)
    return np.array(fs)


ref = run('control: identity (PCA-512, uncentred train)', (1, 0, 0, 0), 0.0025, False)
run('identity, prompt-centred train', (1, 0, 0, 0), 0.0025, True)
res = {}
for l in ((1, 1, 1, 1), (1, 1, 0, 1), (1, 0, 1, 1), (1, 2, 2, 1), (1, 1, 1, 0)):
    for C in (0.0025, 0.01):
        for cen in (True, False):
            res[(l, C, cen)] = run(f'Phi l={l} C={C} {"centred" if cen else "uncentred"} train', l, C, cen)
print('\nbest deltas vs control:')
for k, v in sorted(res.items(), key=lambda kv: -np.mean(kv[1]))[:6]:
    d = v - ref
    print(f'  {str(k):40s} {d.mean():+.4f}  {int((d > 0).sum())}/5')
