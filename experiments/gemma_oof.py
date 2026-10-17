"""Gemma LoRA OOF: standalone scores and marginal value on top of the strong frozen-LLM OOF blend."""
import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score as bacc

ROOT = pathlib.Path(__file__).resolve().parents[1]
src = sys.argv[1] if len(sys.argv) > 1 else 'nb/lora_gemma/full_out/lora_gemma'
tr = pd.read_csv(ROOT / 'data/train.csv')
LABELS = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(LABELS)}).values
pos = {i: k for k, i in enumerate(tr.id.values)}
P = np.full((len(tr), 18), np.nan)
C = np.full((len(tr), 18), np.nan)
fold = np.full(len(tr), -1)
for f in sorted(glob.glob(str(ROOT / src / 'fold*/val_outputs.npz'))):
    z = np.load(f, allow_pickle=True)
    assert list(z['classes']) == LABELS
    r = np.array([pos[i] for i in z['ids']])
    P[r], C[r], fold[r] = z['plain_logits'], z['center_logits_chunk'], int(f.split('fold')[-1][0])
ok = ~np.isnan(P[:, 0])
print('rows with OOF', ok.sum())
cal = lambda L: log_softmax(L * minimize_scalar(lambda t: -log_softmax(L[ok] * t, 1)[np.arange(ok.sum()), y[ok]].mean(),
                                                bounds=(0.05, 20), method='bounded').x, 1)
Lp, Lc = cal(P), cal(C)
print(f'gemma plain {bacc(y[ok], P[ok].argmax(1)):.4f}  centred(chunks2-3) {bacc(y[ok], C[ok].argmax(1)):.4f}  '
      f'half {bacc(y[ok], (0.5 * P + 0.5 * C)[ok].argmax(1)):.4f}')
zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
S = sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
S = log_softmax(S * minimize_scalar(lambda t: -log_softmax(S * t, 1)[np.arange(len(y)), y].mean(),
                                    bounds=(0.1, 20), method='bounded').x, 1)
b = bacc(y[ok], S[ok].argmax(1))
print(f'strong blend {b:.4f}; agreement gemma-vs-strong {np.mean(P[ok].argmax(1) == S[ok].argmax(1)):.3f}')
for wt in (0.25, 0.5, 0.75, 1.0, 1.5):
    B = S + wt * Lp
    fd = [bacc(y[fold == k], B[fold == k].argmax(1)) - bacc(y[fold == k], S[fold == k].argmax(1)) for k in range(5)]
    print(f'strong + {wt:4.2f}*gemma_plain: {bacc(y[ok], B[ok].argmax(1)):.4f} ({bacc(y[ok], B[ok].argmax(1)) - b:+.4f}) '
          f'folds+ {sum(d > 0 for d in fd)}/5')
