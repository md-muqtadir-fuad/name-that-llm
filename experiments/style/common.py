"""Shared data, folds, strong-OOF bases and evaluation for the style-member experiments.

Bases (calibrated log-probs, prompt-disjoint public OOF, scenario '1' = no context):
  FULL  = public q35heads non-centred blend (includes its own TF-IDF 'svm' and 'style' members) -> 0.4915
  NOSVM = same blend without the 'svm' member (keeps 'style'): the analogue of our v1 blend minus stylegram.
A candidate member gives OOF scores S (n x 18). It enters as  base + w * log_softmax(T * S)  (T fit by NLL).
"""
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/style/out'
CACHE = ROOT / 'experiments/style/cache'
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

tr = pd.read_csv(ROOT / 'data/train.csv')
TEXTS = tr.text.astype(str).tolist()
LABELS = sorted(tr.model.unique())
K = len(LABELS)
y = tr.model.map({l: i for i, l in enumerate(LABELS)}).values
g = tr.prompt_id.values
N = len(tr)


def get_folds(split):
    """split 0 = GroupKFold(5) default order (comparable with earlier experiments); 1, 2 = shuffled GroupKFold."""
    if split == 0:
        return list(GroupKFold(5).split(tr, y, g))
    return list(GroupKFold(5, shuffle=True, random_state=1000 + split).split(tr, y, g))


def _calib(L):
    t = minimize_scalar(lambda t: -log_softmax(L * t, 1)[np.arange(N), y].mean(), bounds=(0.01, 50),
                        method='bounded').x
    return log_softmax(L * t, 1), t


def load_bases():
    zz = np.load(ROOT / 'research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=False)
    assert (zz['ids'] == tr.id.values).all()
    names = list(zz['component_names'])
    w = json.loads((ROOT / 'research/ext_oof/q35heads/bundle_manifest.json').read_text())['weights']
    out = {}
    for bname, drop in (('FULL', ()), ('NOSVM', ('svm',))):
        wp = {k: v for k, v in w.items() if 'center' not in k and k not in drop}
        L = sum(v * zz['components'][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
        out[bname] = _calib(L)[0]
    out['_svm'] = zz['components'][names.index('svm'), 0]
    # MID: a weaker 2-member frozen base (Qwen3-8B joint head + DeBERTa joint head), 0.4504
    out['MID'] = _calib((zz['components'][names.index('qwen8_plain_joint'), 0] +
                         zz['components'][names.index('deberta_joint'), 0]) / 2)[0]
    return out


def fit_temp(S):
    """Temperature T minimising NLL of softmax(T*S)."""
    return minimize_scalar(lambda t: -log_softmax(S * t, 1)[np.arange(N), y].mean(), bounds=(1e-3, 200),
                           method='bounded').x


def nll(L):
    return float(-log_softmax(L, 1)[np.arange(N), y].mean())


WGRID = np.round(np.arange(0, 2.51, 0.05), 3)


def _bacc_rows(pred, rows):
    return bacc(y[rows], pred[rows])


def boot_delta(pa, pb, B=2000, seed=0):
    """Prompt-level bootstrap of bacc(pb) - bacc(pa) (paired). Returns (mean, lo95, hi95, P(delta>0))."""
    rng = np.random.default_rng(seed)
    up, gi = np.unique(g, return_inverse=True)
    ca, cb = (pa == y).astype(float), (pb == y).astype(float)
    Y1 = np.eye(K)[y]                                     # n x K
    cnt = rng.multinomial(len(up), np.ones(len(up)) / len(up), size=B)   # B x prompts
    W = cnt[:, gi]                                        # B x n
    den = W @ Y1                                          # B x K
    ra = (W * ca) @ Y1 / np.maximum(den, 1)
    rb = (W * cb) @ Y1 / np.maximum(den, 1)
    d = (rb - ra).mean(1)
    return float(d.mean()), float(np.quantile(d, .025)), float(np.quantile(d, .975)), float((d > 0).mean())


def evaluate(S, name, bases, folds, verbose=True):
    """S: OOF scores (n x K) from `folds`. Returns dict of metrics."""
    S = np.asarray(S, dtype=np.float64)
    T = fit_temp(S)
    lp = log_softmax(S * T, 1)
    res = dict(name=name, alone=round(bacc(y, S.argmax(1)), 4), nll=round(nll(S * T), 4), T=round(T, 4))
    fold_alone = [bacc(y[b], S[b].argmax(1)) for _, b in folds]
    res['alone_fold_sd'] = round(float(np.std(fold_alone)), 4)
    for bname in ('NOSVM', 'FULL', 'MID'):
        Lb = bases[bname]
        b0 = bacc(y, Lb.argmax(1))
        curve = np.array([bacc(y, (Lb + w * lp).argmax(1)) for w in WGRID])
        nlls = np.array([nll(Lb + w * lp) for w in WGRID])
        wb = WGRID[curve.argmax()]
        wn = WGRID[nlls.argmin()]
        # nested: choose w on the other 4 folds (by NLL, the smooth criterion), apply to the held-out fold
        nested_pred = np.zeros(N, dtype=int)
        fold_d = []
        for _, b in folds:
            rest = np.setdiff1d(np.arange(N), b)
            nl = [float(-log_softmax(Lb[rest] + w * lp[rest], 1)[np.arange(len(rest)), y[rest]].mean()) for w in WGRID]
            wk = WGRID[int(np.argmin(nl))]
            nested_pred[b] = (Lb[b] + wk * lp[b]).argmax(1)
            fold_d.append(bacc(y[b], nested_pred[b]) - bacc(y[b], Lb[b].argmax(1)))
        nested = bacc(y, nested_pred)
        bm, lo, hi, pp = boot_delta(Lb.argmax(1), nested_pred)
        res.update({f'{bname}_base': round(b0, 4), f'{bname}_wNLL': wn, f'{bname}_dNLLw': round(curve[list(WGRID).index(wn)] - b0, 4),
                    f'{bname}_wbest': wb, f'{bname}_dbest': round(curve.max() - b0, 4),
                    f'{bname}_nested_d': round(nested - b0, 4),
                    f'{bname}_nested_fold_d': [round(x, 4) for x in fold_d],
                    f'{bname}_nested_folds_pos': int(sum(x > 0 for x in fold_d)),
                    f'{bname}_boot': (round(lo, 4), round(hi, 4), round(pp, 3)),
                    f'{bname}_blend_nll_gain': round(nll(Lb) - nlls.min(), 4),
                    f'{bname}_curve': {float(w): round(c - b0, 4) for w, c in zip(WGRID[::4], curve[::4])}})
    if verbose:
        print(f"{name:48s} alone {res['alone']:.4f}  nll {res['nll']:.3f} | NOSVM nested {res['NOSVM_nested_d']:+.4f} "
              f"(w~{res['NOSVM_wNLL']}, best {res['NOSVM_dbest']:+.4f}@{res['NOSVM_wbest']}, folds+ {res['NOSVM_nested_folds_pos']}/5, "
              f"CI {res['NOSVM_boot'][0]:+.3f}..{res['NOSVM_boot'][1]:+.3f}) | FULL nested {res['FULL_nested_d']:+.4f} "
              f"(w~{res['FULL_wNLL']}, best {res['FULL_dbest']:+.4f}, folds+ {res['FULL_nested_folds_pos']}/5) | MID nested {res['MID_nested_d']:+.4f}", flush=True)
    return res


def save_oof(name, split, S):
    np.save(CACHE / f'oof_{name}_s{split}.npy', np.asarray(S, dtype=np.float32))


def load_oof(name, split):
    p = CACHE / f'oof_{name}_s{split}.npy'
    return np.load(p) if p.exists() else None


def log_result(res, split, path=OUT / 'results.jsonl'):
    r = dict(res)
    r['split'] = split
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(r, default=float) + '\n')


def evaluate_multi(Ss, name, bases, folds, verbose=True):
    """Several members at once: base + sum_j w_j * log_softmax(T_j S_j), w_j >= 0 fit by NLL on 4 folds
    (L-BFGS-B), applied to the held-out fold. Returns nested deltas per base."""
    from scipy.optimize import minimize
    lps = [log_softmax(np.asarray(S, dtype=np.float64) * fit_temp(np.asarray(S, dtype=np.float64)), 1) for S in Ss]
    Lst = np.stack(lps)                                       # m x n x K
    res = dict(name=name)
    for bname in ('NOSVM', 'FULL', 'MID'):
        Lb = bases[bname]
        b0 = bacc(y, Lb.argmax(1))
        pred = np.zeros(N, dtype=int)
        fold_d, ws = [], []
        for _, b in folds:
            rest = np.setdiff1d(np.arange(N), b)

            def f(w):
                Z = Lb[rest] + np.tensordot(w, Lst[:, rest], 1)
                return float(-log_softmax(Z, 1)[np.arange(len(rest)), y[rest]].mean())
            w = minimize(f, np.full(len(Ss), 0.1), method='L-BFGS-B', bounds=[(0, 3)] * len(Ss)).x
            ws.append(w)
            pred[b] = (Lb[b] + np.tensordot(w, Lst[:, b], 1)).argmax(1)
            fold_d.append(bacc(y[b], pred[b]) - bacc(y[b], Lb[b].argmax(1)))
        bm, lo, hi, pp = boot_delta(Lb.argmax(1), pred)
        res.update({f'{bname}_nested_d': round(bacc(y, pred) - b0, 4), f'{bname}_nested_fold_d': [round(x, 4) for x in fold_d],
                    f'{bname}_nested_folds_pos': int(sum(x > 0 for x in fold_d)), f'{bname}_boot': (round(lo, 4), round(hi, 4), round(pp, 3)),
                    f'{bname}_w': [round(float(x), 3) for x in np.mean(ws, 0)]})
    if verbose:
        print(f"{name:48s} | NOSVM nested {res['NOSVM_nested_d']:+.4f} w {res['NOSVM_w']} folds+ {res['NOSVM_nested_folds_pos']}/5 "
              f"| FULL nested {res['FULL_nested_d']:+.4f} folds+ {res['FULL_nested_folds_pos']}/5 | MID nested {res['MID_nested_d']:+.4f} w {res['MID_w']}", flush=True)
    return res
