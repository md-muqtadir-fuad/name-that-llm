"""Lab-gated mixture of experts vs a flat 18-way head, on frozen Qwen3-8B L7+22 (5-fold GroupKFold by prompt).
log p(y|x) = g * log p(lab(y)|x) + log p(y | lab(y), x)
gate: 6-way lab LR; experts: per-lab LR trained only on that lab's rows, standardised WITHIN the lab (training-fold
rows of that lab; test rows standardised with the same within-lab train stats after global test-standardisation).
Also: flat + expert blend (residual MoE): log p_flat + e * log p(y|lab,x)."""
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
LABELS = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(LABELS)}).values
g = tr.prompt_id.values
labs = sorted({l[:3] for l in LABELS})
lab_of = np.array([labs.index(l[:3]) for l in LABELS])
yl = lab_of[y]
z = np.load(ROOT / 'experiments/papers/qwen3_8b_train_feats.npz')
lays = list(z['layers'])
X = np.concatenate([z['E'][:, lays.index(7)], z['E'][:, lays.index(22)]], 1).astype(np.float32)
folds = list(GroupKFold(5).split(X, y, g))
T0 = time.time()
K = len(LABELS)
LF = np.zeros((len(y), K))       # flat
LG = np.zeros((len(y), len(labs)))  # gate
LE = {c: np.full((len(y), K), -50.0) for c in (0.0025, 0.01, 0.04)}  # expert log p(y|lab,x), -50 outside lab
for a, b in folds:
    sc = StandardScaler().fit(X[a])
    A = sc.transform(X[a])
    B = (X[b] - X[b].mean(0)) / (X[b].std(0) + 1e-6)
    LF[b] = LogisticRegression(C=0.0025, max_iter=3000).fit(A, y[a]).predict_log_proba(B)
    LG[b] = LogisticRegression(C=0.0025, max_iter=3000).fit(A, yl[a]).predict_log_proba(B)
    for li in range(len(labs)):
        cls = np.where(lab_of == li)[0]
        if len(cls) == 1:
            for c in LE:
                LE[c][np.ix_(b, cls)] = 0.0
            continue
        ra = a[np.isin(y[a], cls)]
        s2 = StandardScaler().fit(A[np.isin(y[a], cls)])          # within-lab standardisation
        Bw = s2.transform(B)
        for c in LE:
            m = LogisticRegression(C=c, max_iter=3000).fit(s2.transform(A[np.isin(y[a], cls)]), y[ra])
            LE[c][np.ix_(b, m.classes_)] = m.predict_log_proba(Bw)
    print(f'fold done {time.time() - T0:.0f}s', flush=True)


def rep(name, L):
    fs = [bacc(y[b], L[b].argmax(1)) for _, b in folds]
    print(f'{name:48s} {bacc(y, L.argmax(1)):.4f}  folds {np.round(fs, 3)}', flush=True)
    return np.array(fs)


base = rep('flat 18-way', LF)
print(f'gate lab acc {bacc(yl, LG.argmax(1)):.4f}  flat->lab acc {bacc(yl, lab_of[LF.argmax(1)]):.4f}')
for li, ln in enumerate(labs):
    m = yl == li
    if (lab_of == li).sum() > 1:
        print(f'  within-{ln}: flat {bacc(y[m], LF[m][:, lab_of == li].argmax(1)):.3f}  '
              + '  '.join(f'expC{c} {bacc(y[m], LE[c][m][:, lab_of == li].argmax(1)):.3f}' for c in LE))
out = {}
for c in LE:
    for gw in (1.0, 2.0):
        out[f'MoE C{c} gate{gw}'] = rep(f'MoE: {gw}*log p(lab) + expert C={c}', gw * LG[:, lab_of] + LE[c])
    for e in (0.5, 1.0):
        out[f'flat+{e}*exp C{c}'] = rep(f'flat + {e}*expert C={c} (residual MoE)', LF + e * np.maximum(LE[c], -50))
print('\ndelta vs flat (mean, positive folds):')
for k, v in out.items():
    d = v - base
    print(f'  {k:28s} {d.mean():+.4f}  {int((d > 0).sum())}/5')
