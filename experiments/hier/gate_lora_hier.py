"""Falsification tests of two claims from an external review (2026-10-19), on prompt-grouped OOF (folds 0-2, where the
Gemma-ext and GLM-ext LoRA OOF exist):
  A) "the within-family gates (argmax_G log P + lam * SVC decision) erase the LoRAs' within-family gains":
     strong blend +/- LoRAs, then gates at lam in {0, 0.25, 0.5, 1.0, 1.5} (gate decisions = OOF from
     experiments/gates/gate_test.py, with external data: dec_<lab>_1.npy).
  B) "sibling vote-splitting under argmax": group-first decoding with within-group sharpening
     P~(c) = P(G) * P(c|G)^gamma / sum_j P(j|G)^gamma for sibling groups, gamma in {1, 1.25, 1.5, 2, 3}.
Writes experiments/hier/out/results.csv and paper/experiments.csv rows E040_* (idempotent)."""
import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score as bacc

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
from log import log_row  # noqa: E402

OUT = ROOT / 'experiments/hier/out'
OUT.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(ROOT / 'data/train.csv')
L = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(L)}).values
pos = {i: k for k, i in enumerate(tr.id.values)}


def lora(src):
    Pm, fold = np.full((len(tr), 18), np.nan), np.full(len(tr), -1)
    for f in sorted(glob.glob(str(ROOT / src / 'fold*/val_outputs.npz'))):
        z = np.load(f, allow_pickle=True)
        r = np.array([pos[i] for i in z['ids']])
        Pm[r], fold[r] = z['plain_logits'], int(f.split('fold')[-1][0])
    return Pm, fold


G, fold = lora('nb/lora_gemma_ext/full_out/lora_gemma')
M, _ = lora('nb/lora_glm_ext/full_out/lora_gemma')
ok = fold >= 0
yy, ff = y[ok], fold[ok]
cal = lambda Z: log_softmax(Z * minimize_scalar(lambda t: -log_softmax(Z * t, 1)[np.arange(len(Z)), yy].mean(),
                                                bounds=(0.05, 50), method='bounded').x, 1)
zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
wp = {k: v for k, v in w.items() if 'center' not in k}
S = cal((sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values()))[ok])
Gc, Mc = cal(G[ok]), cal(M[ok])
DEC = {lab: np.load(ROOT / f'experiments/gates/dec_{lab}_1.npy')[ok] for lab in ('claude', 'gemini', 'gpt')}
GIDX = {lab: np.array([i for i, l in enumerate(L) if l.startswith(lab)]) for lab in DEC}
SIB = {'gemini_flash': ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],
       'gpt56': ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'],
       'claude_opus': ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5'],
       'glm': ['glm-5.3', 'glm-5.3-flash']}
SIB = {k: np.array([L.index(m) for m in v]) for k, v in SIB.items()}


def gates(Z, lam, labs=('claude', 'gemini', 'gpt')):
    pred = Z.argmax(1).copy()
    for lab in labs:
        g = GIDX[lab]
        m = np.isin(pred, g)
        new = g[(Z[:, g] + lam * DEC[lab]).argmax(1)]
        pred[m] = new[m]
    return pred


def sharpen(Z, gamma):
    P = np.exp(Z - Z.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    for g in SIB.values():
        pg = P[:, g].sum(1, keepdims=True)
        q = (P[:, g] / pg) ** gamma
        P[:, g] = pg * q / q.sum(1, keepdims=True)
    return np.log(P + 1e-12)


def score(pred, ref_pred):
    tot = bacc(yy, pred)
    fd = [bacc(yy[ff == k], pred[ff == k]) - bacc(yy[ff == k], ref_pred[ff == k]) for k in np.unique(ff)]
    return round(tot, 4), sum(d > 0 for d in fd), [round(d, 4) for d in fd]


res = []
base_pred = S.argmax(1)
LO = S + 0.5 * Gc + 1.0 * Mc          # best LoRA mix from E031
for name, Z in [('strong', S), ('strong+loras', LO)]:
    ref = Z.argmax(1)
    for lam in (0.0, 0.25, 0.5, 1.0, 1.5):
        p = gates(Z, lam) if lam > 0 else ref
        t, fp, fd = score(p, ref)
        res.append(dict(test='A_gates', base=name, param=f'lam={lam}', bacc=t, delta_vs_same_base_no_gate=round(t - bacc(yy, ref), 4),
                        folds_pos=f'{fp}/3', fold_deltas=json.dumps(fd), delta_vs_strong=round(t - bacc(yy, base_pred), 4)))
for name, Z in [('strong', S), ('strong+loras', LO)]:
    ref = Z.argmax(1)
    for gamma in (1.0, 1.25, 1.5, 2.0, 3.0):
        p = sharpen(Z, gamma).argmax(1)
        t, fp, fd = score(p, ref)
        res.append(dict(test='B_sharpen', base=name, param=f'gamma={gamma}', bacc=t, delta_vs_same_base_no_gate=round(t - bacc(yy, ref), 4),
                        folds_pos=f'{fp}/3', fold_deltas=json.dumps(fd), delta_vs_strong=round(t - bacc(yy, base_pred), 4)))
# best combination: loras + sharpen + gates
for gamma in (1.5, 2.0):
    for lam in (0.25, 0.5, 1.5):
        Z = sharpen(LO, gamma)
        p = gates(Z, lam)
        t, fp, fd = score(p, LO.argmax(1))
        res.append(dict(test='AB_combo', base='strong+loras', param=f'gamma={gamma},lam={lam}', bacc=t,
                        delta_vs_same_base_no_gate=round(t - bacc(yy, LO.argmax(1)), 4), folds_pos=f'{fp}/3',
                        fold_deltas=json.dumps(fd), delta_vs_strong=round(t - bacc(yy, base_pred), 4)))
R = pd.DataFrame(res)
R.to_csv(OUT / 'results.csv', index=False)
print(R.drop(columns=['fold_deltas']).to_string(index=False))
ex = pd.read_csv(ROOT / 'paper/experiments.csv', dtype=str, keep_default_na=False)
ex[~ex['id'].str.startswith('E040_')].to_csv(ROOT / 'paper/experiments.csv', index=False)
for r in res:
    log_row(id=f"E040_{r['test']}_{r['base']}_{r['param']}", category='postproc',
            description=f"review falsification: {r['test']} on {r['base']} ({r['param']}), OOF folds 0-2",
            config='experiments/hier/gate_lora_hier.py', metric='oof_bacc_folds0-2', baseline='same base without this step',
            value=str(r['bacc']), delta=str(r['delta_vs_same_base_no_gate']), folds_pos=r['folds_pos'],
            files='experiments/hier/out/results.csv', notes=f"fold deltas {r['fold_deltas']}; vs strong {r['delta_vs_strong']}")
