"""E043: 'orthographic shield' claim (Gemini 3.8 review): Claude & GLM never use curly quotes/apostrophes, Gemini never
uses em-dashes in train. 1) verify the rates per label; 2) OOF test of the penalty (-2.0 on Claude+GLM log-probs if the
passage has a curly quote/apostrophe; on Gemini if it has >= 2 em-dashes) on strong and strong+LoRAs.
Writes experiments/hier/out/shield_rates.csv, shield_test.csv; paper/experiments.csv E043_* (idempotent)."""
import glob, json, pathlib, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper'))
from log import log_row
OUT = ROOT / 'experiments/hier/out'
tr = pd.read_csv(ROOT / 'data/train.csv'); L = sorted(tr.model.unique()); y = tr.model.map({l: i for i, l in enumerate(L)}).values
curly = tr.text.str.contains('[’‘“”]').values
em2 = (tr.text.str.count('—') >= 2).values
em1 = tr.text.str.contains('—').values
R = pd.DataFrame({'label': L, 'n': np.bincount(y), 'share_curly': [curly[y == k].mean() for k in range(18)],
                  'share_emdash_any': [em1[y == k].mean() for k in range(18)], 'share_emdash_2plus': [em2[y == k].mean() for k in range(18)]}).round(4)
R.to_csv(OUT / 'shield_rates.csv', index=False); print(R.to_string(index=False))
fold = np.zeros(len(tr), int)
for k, (_, b) in enumerate(GroupKFold(5).split(tr, y, tr.prompt_id)): fold[b] = k
def cal(Z, m):
    t = minimize_scalar(lambda t: -log_softmax(Z[m] * t, 1)[np.arange(m.sum()), y[m]].mean(), bounds=(0.05, 50), method='bounded').x
    return log_softmax(Z * t, 1)
zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz'); names = list(zz['component_names'])
w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']; wp = {k: v for k, v in w.items() if 'center' not in k}
ALL = np.ones(len(y), bool); S = cal(sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values()), ALL)
pos = {i: k for k, i in enumerate(tr.id.values)}
def lora(src):
    Z = np.full((len(tr), 18), np.nan)
    for f in sorted(glob.glob(str(ROOT / src / 'fold*/val_outputs.npz'))):
        z = np.load(f, allow_pickle=True); Z[np.array([pos[i] for i in z['ids']])] = z['plain_logits']
    return Z
G, M = lora('nb/lora_gemma_ext/full_out/lora_gemma'), lora('nb/lora_glm_ext/full_out/lora_gemma'); OK3 = ~np.isnan(G[:, 0])
LO = np.full_like(S, np.nan); LO[OK3] = S[OK3] + 0.5 * cal(np.nan_to_num(G), OK3)[OK3] + cal(np.nan_to_num(M), OK3)[OK3]
CG = np.array([l.startswith(('claude', 'glm')) for l in L]); GEM = np.array([l.startswith('gemini') for l in L])
res = []
for name, Z, rows in [('strong', S, ALL), ('strong+loras', LO, OK3)]:
    for pen in (0.5, 1.0, 2.0):
        Zs = Z - pen * (curly[:, None] * CG[None] + em2[:, None] * GEM[None])
        used = sorted(set(fold[rows]))
        fd = [bacc(y[rows & (fold == k)], Zs[rows & (fold == k)].argmax(1)) - bacc(y[rows & (fold == k)], Z[rows & (fold == k)].argmax(1)) for k in used]
        viol = int((np.isin(Z[rows].argmax(1), np.where(CG)[0]) & curly[rows]).sum() + (np.isin(Z[rows].argmax(1), np.where(GEM)[0]) & em2[rows]).sum())
        r = dict(base=name, penalty=pen, bacc=round(bacc(y[rows], Zs[rows].argmax(1)), 4), delta=round(bacc(y[rows], Zs[rows].argmax(1)) - bacc(y[rows], Z[rows].argmax(1)), 4),
                 folds_pos=f'{sum(d > 0 for d in fd)}/{len(fd)}', violating_predictions_before=viol, rows=int(rows.sum()))
        res.append(r); print(r)
pd.DataFrame(res).to_csv(OUT / 'shield_test.csv', index=False)
ex = pd.read_csv(ROOT / 'paper/experiments.csv', dtype=str, keep_default_na=False); ex[~ex['id'].str.startswith('E043_')].to_csv(ROOT / 'paper/experiments.csv', index=False)
for r in res:
    log_row(id=f"E043_{r['base']}_pen{r['penalty']}", category='postproc', description=f"orthographic shield (curly->not Claude/GLM, 2+ em-dash->not Gemini) on {r['base']}",
            config='experiments/hier/shield_test.py', metric='oof_bacc', baseline='same base', value=str(r['bacc']), delta=str(r['delta']), folds_pos=r['folds_pos'],
            files='experiments/hier/out/shield_test.csv;experiments/hier/out/shield_rates.csv', notes=f"violating predictions before: {r['violating_predictions_before']} of {r['rows']}")
