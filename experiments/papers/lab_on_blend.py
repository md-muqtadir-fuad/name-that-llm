"""Does the hierarchical lab term still help on top of the strong blend OOF? Lab LR on frozen Qwen3-8B L7+22 (test-std),
prompt-disjoint OOF; strong = public prompt-disjoint frozen-LLM blend OOF (calibrated). Reports per-fold deltas."""
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
LAB = {'gpt': 0, 'gem': 1, 'cla': 2, 'glm': 3, 'qwe': 4, 'kim': 5}
lab_of = np.array([LAB[l[:3]] for l in LABELS])

zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
L = sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
t = minimize_scalar(lambda t: -log_softmax(L * t, 1)[np.arange(len(y)), y].mean(), bounds=(0.1, 20), method='bounded').x
L = log_softmax(L * t, 1)

z = np.load(ROOT / 'experiments/papers/qwen3_8b_train_feats.npz')
lays = list(z['layers'])
X = np.concatenate([z['E'][:, lays.index(7)], z['E'][:, lays.index(22)]], 1).astype(np.float32)
folds = list(GroupKFold(5).split(tr, y, g))
Llab = np.zeros((len(y), 6))
for a, b in folds:
    sc = StandardScaler().fit(X[a])
    c = LogisticRegression(C=0.0025, max_iter=3000).fit(sc.transform(X[a]), lab_of[y[a]])
    Llab[b] = c.predict_log_proba((X[b] - X[b].mean(0)) / (X[b].std(0) + 1e-6))
base = bacc(y, L.argmax(1))
print(f'strong blend OOF {base:.4f}; lab head alone lab-acc {bacc(lab_of[y], Llab.argmax(1)):.4f}, '
      f'blend lab-acc {bacc(lab_of[y], lab_of[L.argmax(1)]):.4f}')
for beta in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
    L2 = L + beta * Llab[:, lab_of]
    fd = [bacc(y[b], L2[b].argmax(1)) - bacc(y[b], L[b].argmax(1)) for _, b in folds]
    print(f'beta {beta:4.2f}: {bacc(y, L2.argmax(1)):.4f} delta {bacc(y, L2.argmax(1)) - base:+.4f} '
          f'folds+ {sum(d > 0 for d in fd)}/5 min {min(fd):+.4f}')
np.save(ROOT / 'experiments/papers/out/Llab_oof.npy', Llab)
