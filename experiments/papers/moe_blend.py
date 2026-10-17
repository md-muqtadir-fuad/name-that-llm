"""Does the lab-gated MoE (gate 1.0, experts C=0.0025, within-lab standardisation) beat the flat head ON TOP OF the
strong blend? Compare strong + w*flat vs strong + w*MoE (same Qwen3-8B L7+22 features, same folds)."""
import json
import pathlib

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
labs = sorted({l[:3] for l in LABELS})
lab_of = np.array([labs.index(l[:3]) for l in LABELS])
z = np.load(ROOT / 'experiments/papers/qwen3_8b_train_feats.npz')
lays = list(z['layers'])
X = np.concatenate([z['E'][:, lays.index(7)], z['E'][:, lays.index(22)]], 1).astype(np.float32)
folds = list(GroupKFold(5).split(X, y, g))
LF, LG, LE = np.zeros((len(y), 18)), np.zeros((len(y), 6)), np.zeros((len(y), 18))
for a, b in folds:
    sc = StandardScaler().fit(X[a])
    A, B = sc.transform(X[a]), (X[b] - X[b].mean(0)) / (X[b].std(0) + 1e-6)
    LF[b] = LogisticRegression(C=0.0025, max_iter=3000).fit(A, y[a]).predict_log_proba(B)
    LG[b] = LogisticRegression(C=0.0025, max_iter=3000).fit(A, lab_of[y[a]]).predict_log_proba(B)
    for li in range(6):
        cls = np.where(lab_of == li)[0]
        if len(cls) == 1:
            continue
        m = np.isin(y[a], cls)
        s2 = StandardScaler().fit(A[m])
        e = LogisticRegression(C=0.0025, max_iter=3000).fit(s2.transform(A[m]), y[a][m])
        LE[np.ix_(b, e.classes_)] = e.predict_log_proba(s2.transform(B))
LM = LG[:, lab_of] + LE
LM = LM - np.log(np.exp(LM).sum(1, keepdims=True))
np.savez(ROOT / 'experiments/papers/out/moe_oof.npz', LF=LF, LM=LM)
zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
S = sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
S = log_softmax(S * minimize_scalar(lambda t: -log_softmax(S * t, 1)[np.arange(len(y)), y].mean(),
                                    bounds=(0.1, 20), method='bounded').x, 1)
base = bacc(y, S.argmax(1))
print(f'flat alone {bacc(y, LF.argmax(1)):.4f}  MoE alone {bacc(y, LM.argmax(1)):.4f}  strong {base:.4f}')
for wt in (0.25, 0.5, 1.0):
    for nm, L in (('flat', LF), ('MoE ', LM)):
        B2 = S + wt * L
        fd = [bacc(y[b], B2[b].argmax(1)) - bacc(y[b], S[b].argmax(1)) for _, b in folds]
        print(f'strong + {wt:4.2f}*{nm}: {bacc(y, B2.argmax(1)):.4f} ({bacc(y, B2.argmax(1)) - base:+.4f}) '
              f'folds+ {sum(d > 0 for d in fd)}/5')
