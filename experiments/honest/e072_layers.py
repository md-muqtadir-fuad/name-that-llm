"""E072 (local part): layer-by-layer value of frozen Phi-4-14B and OLMo-2-13B, with frozen Qwen3-8B-Base as reference.
Input: research/layer_screen/oof_<model>.npy (n_layers+1, 2699, 18) honest OOF log-probs of a PCA-256 + LR probe per
layer (Kaggle CPU kernel ntl-layer-screen-eval, seeded 5-fold split = the honest-OOF harness split).
Per layer: (a) standalone balanced accuracy; (b) gain when the probe joins the honest blend of the 0.664 members we can
score (q14v5/q8v5/ll8v5 plain+twin+Gaussian, tfidf/distort/svc4 at the hand weights, Sinkhorn 0.45): weight picked from
{0.5, 1, 2, 3} on 4 seeded folds and scored on the 5th, 6 draws of test-like 2-3 answer groups.
Run from the repo root: python experiments/honest/e072_layers.py"""
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bas

sys.path.insert(0, 'paper')
from log import log_row  # noqa: E402

src = open('experiments/honest/e069_bias.py', encoding='utf-8').read()
exec(src.split("rng = np.random.default_rng(0)\nres = []")[0])   # y, fold, pid, blend(), sinkhorn()

OOF = {f.split('oof_')[1][:-4]: np.load(f).astype(np.float64) for f in sorted(glob.glob('research/layer_screen/oof_*.npy'))}
print({k: v.shape for k, v in OOF.items()})
WGRID = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)   # v2: includes 0 (no probe); v1 forced >= 0.5, where every probe hurts
rng = np.random.default_rng(0)
bases = []
for d in range(6):
    g = np.full(len(y), -1); nxt = 0
    for p_ in np.unique(pid):
        ii = rng.permutation(np.flatnonzero(pid == p_)); i = 0
        while i < len(ii):
            n = 2 if len(ii) - i in (2, 4) else int(rng.integers(2, 4)); g[ii[i:i + n]] = nxt; nxt += 1; i += n
    bases.append(blend(g, False))   # lpn = sum(w * lp) / n_members (n = 11)
NB = 11
res = []
for name, O in OOF.items():
    for l in range(O.shape[0]):
        sa = bas(y, O[l].argmax(1))
        gains = []
        for lpn in bases:
            tot = lpn * NB
            pred = np.empty(len(y), int)
            for k in range(5):
                tr, te = fold != k, fold == k
                sc = {w: bas(y[tr], sinkhorn((tot[tr] + w * O[l][tr]) / (NB + 1)).argmax(1)) for w in WGRID}
                w = max(sc, key=sc.get)
                pred[te] = sinkhorn((tot[te] + w * O[l][te]) / (NB + 1)).argmax(1)
            # Sinkhorn per fold slice above uses each fold's own marginal; score the base the same way
            bpred = np.empty(len(y), int)
            for k in range(5):
                te = fold == k
                bpred[te] = sinkhorn(lpn[te]).argmax(1)
            gains.append(bas(y, pred) - bas(y, bpred))
        res.append(dict(model=name, layer=l, rel_depth=round(l / (O.shape[0] - 1), 3), standalone=sa,
                        gain=np.mean(gains), se=np.std(gains) / np.sqrt(len(gains)), better=f'{sum(x > 0 for x in gains)}/{len(gains)}'))
    best = max((r for r in res if r['model'] == name), key=lambda r: r['gain'])
    print(f"{name}: best standalone {max(r['standalone'] for r in res if r['model'] == name):.4f} | best blend gain "
          f"layer {best['layer']} ({best['rel_depth']}) {best['gain']:+.4f} (se {best['se']:.4f}, {best['better']})", flush=True)
R = pd.DataFrame(res)
R.to_csv('experiments/honest/out/e072_layers.csv', index=False)
print(R.round(4).to_string(index=False))
summ = R.loc[R.groupby('model').gain.idxmax()]
log_row(id='E072', category='backbone', metric='bacc_layer_probe_and_blend_gain',
        description='layer-by-layer screen: frozen Phi-4-14B, OLMo-2-1124-13B (8-bit) and Qwen3-8B-Base (fp16, reference); every layer mean-pooled, PCA-256 + LR probe; standalone and gain on the honest 0.664 blend',
        config='seeded 5-fold OOF (Kaggle CPU ntl-layer-screen-eval); blend gain: weight in {0,0.05,0.1,0.2,0.35,0.5} picked on 4 folds, Sinkhorn 0.45 per fold slice, 6 draws of 2-3 groups',
        baseline='0 (blend without the probe)', value='; '.join(f"{r.model} L{r.layer} {r.gain:+.4f}" for r in summ.itertuples()),
        folds_pos='; '.join(f"{r.model} {r.better}" for r in summ.itertuples()), files='experiments/honest/out/e072_layers.csv',
        notes='; '.join(f"{m}: best standalone {R[R.model == m].standalone.max():.4f} at L{int(R[R.model == m].standalone.idxmax() - R[R.model == m].index[0])}"
                        for m in R.model.unique()))
