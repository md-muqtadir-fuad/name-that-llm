"""E071: honest test of the 0.664 typography likelihood (V31/V38 block in infer.py) and its strength.
Exact replica of the block (37 marks, hierarchical Beta-Bernoulli rates kappa 24, marker weights clipped 0.25-2, evidence
centred, clipped, gated by confidence (margin < 0.6), length and marker coverage), refit per seeded fold on the other
folds' train rows only, applied after Sinkhorn 0.45 to the honest blend of the members we can score (0.664 hand weights,
test-like 2-3 answer groups). Grid: scale {0, 0.75, 1.2, 1.5, 2.5, 4} x clip {0.3, 0.6}. The 0.664 uses scale 1.5,
clip 0.3. Run from the repo root: python experiments/honest/e071_typography.py"""
import ast
import re
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score as bas

sys.path.insert(0, 'paper')
from log import log_row  # noqa: E402

src = open('experiments/honest/e069_bias.py', encoding='utf-8').read()
exec(src.split("rng = np.random.default_rng(0)\nres = []")[0])   # loads y, fold, pid, train, blend(), sinkhorn()

line = next(l for l in open('nb/honest_cpu/infer_0664.py', encoding='utf-8') if l.strip().startswith('_marks = ['))
MARKS = ast.literal_eval(line.split('=', 1)[1].strip())
assert len(MARKS) == 37, len(MARKS)
text = train.text.fillna('').astype(str).to_numpy()
X = np.array([[float(m in t) for m in MARKS] for t in text], dtype=np.float32)
WC = np.array([len(re.findall(r'\b\w+\b', t)) for t in text], dtype=np.float32)


def evidence_for(k, scale, clip):
    tr, te = fold != k, fold == k
    Xtr, ytr, Xte = X[tr], y[tr], X[te]
    freq = np.bincount(ytr, minlength=18).astype(np.float64)
    hits = np.stack([Xtr[ytr == c].sum(0) for c in range(18)]).astype(np.float64)
    g = (Xtr.sum(0) + 2.0) / (len(Xtr) + 4.0)
    rate = np.clip((hits + 24.0 * g[None]) / (freq[:, None] + 24.0), 0.01, 0.99)
    g = np.clip(g, 0.01, 0.99)
    sig = np.sqrt(np.mean((rate - g[None]) ** 2, 0))
    fw = np.clip(sig / max(float(np.median(sig[sig > 0])), 1e-6), 0.25, 2.0)
    ev = (Xte * fw) @ np.log(rate).T + ((1 - Xte) * fw) @ np.log1p(-rate).T
    ev -= (((Xte * fw) @ np.log(g)) + (((1 - Xte) * fw) @ np.log1p(-g)))[:, None]
    ev -= ev.mean(1, keepdims=True)
    return np.clip(scale * ev / max(float(fw.sum()), 1e-6), -clip, clip)


def apply_typo(tot, scale, clip):
    out = tot.copy()
    P = softmax(tot, 1); s = np.sort(P, 1); margin = s[:, -1] - s[:, -2]
    gate = np.clip((0.60 - margin) / 0.60, 0, 1) * np.clip(np.sqrt(np.maximum(WC, 1) / 80.0), 0.40, 1.0) \
        * np.clip(np.sqrt((X.sum(1) + 1.0) / 4.0), 0.45, 1.0)
    for k in range(5):
        te = fold == k
        out[te] += gate[te, None] * evidence_for(k, scale, clip)
    return out


rng = np.random.default_rng(0)
GRID = [(0.0, 0.3)] + [(s, c) for s in (0.75, 1.2, 1.5, 2.5, 4.0) for c in (0.3, 0.6)]
res = []
for d in range(8):
    g = np.full(len(y), -1); nxt = 0
    for p_ in np.unique(pid):
        ii = rng.permutation(np.flatnonzero(pid == p_)); i = 0
        while i < len(ii):
            n = 2 if len(ii) - i in (2, 4) else int(rng.integers(2, 4)); g[ii[i:i + n]] = nxt; nxt += 1; i += n
    tot = sinkhorn(blend(g, False))
    for sc, cl in GRID:
        res.append(dict(draw=d, scale=sc, clip=cl, bacc=bas(y, apply_typo(tot, sc, cl).argmax(1))))
R = pd.DataFrame(res)
P = R.pivot_table(index='draw', columns=['scale', 'clip'], values='bacc')
base = P[(0.0, 0.3)]
rows = []
for col in P.columns:
    dlt = P[col] - base
    rows.append(dict(scale=col[0], clip=col[1], bacc=P[col].mean(), diff=dlt.mean(), se=dlt.std() / np.sqrt(len(dlt)), better=f'{(dlt > 0).sum()}/{len(dlt)}'))
S = pd.DataFrame(rows)
print(S.round(4).to_string(index=False))
R.to_csv('experiments/honest/out/e071_typography.csv', index=False)
cur = S[np.isclose(S['scale'], 1.5) & np.isclose(S['clip'], 0.3)].iloc[0]
best = S.iloc[S['diff'].idxmax()]
log_row(id='E071', category='postprocess', metric='bacc_sim_seeded_5fold',
        description='0.664 typography likelihood (37 marks, Beta-Bernoulli, gated) refit per seeded fold, applied after Sinkhorn 0.45 to the honest blend; scale x clip grid',
        config='8 draws of 2-3 groups; members q14v5/q8v5/ll8v5 (plain+twin+Gaussian) + tfidf/distort/svc4 at 0.664 hand weights; no Gemma',
        baseline=f'{base.mean():.4f}', value=f"{cur['bacc']:.4f}", folds_pos=cur['better'], files='experiments/honest/out/e071_typography.csv',
        notes=f"0.664 setting (1.5, 0.3) {cur['diff']:+.4f} (se {cur['se']:.4f}); best ({best['scale']}, {best['clip']}) {best['diff']:+.4f} (se {best['se']:.4f}, {best['better']})")
