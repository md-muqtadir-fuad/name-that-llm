"""E066: honest test-like simulation of the public 0.664 blend (the members we can score honestly) + NNLS weights,
Sinkhorn strength under class-mix shift, and Gemma as an extra member.
Inputs (all per-row honest OOF, train order):
  research/honest_oof/gpu/honest_oof_q8_ll8.npz (+ honest_oof_q14.npz when present): fold-adapter logits, centred-head
    projections, Gaussian-head U/m/lam  (kernel ntl-honest-oof, seeded split)
  research/honest_oof/cpu_members_oof.npz: tfidf, distort, svc4 log-probs (E065)
  Gemma-4-12B two-stage OOF (our GroupKFold folds 0-2 only): plain + chunk-centred logits
Test-like structure: per draw, each prompt's answers are cut into random groups of 2-3; the twin (centred) and Gaussian
heads see only their group. Members the 0.664 blend also has but that cannot be scored honestly (n14b, q25_14, deb,
ATLAS, comp stack, typography) are left out, so absolute numbers are for this subset only.
Blend = infer.py: tot = sum(w * lp); P = softmax(tot / n_members); damped Sinkhorn to 0.65 * pred marginal + 0.35 * prior.
Run from the repo root: python experiments/honest/stack_sim.py"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import log_softmax, softmax
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.model_selection import GroupKFold

sys.path.insert(0, 'nb/honest_oof')
import gauss as ga  # noqa: E402

sys.path.insert(0, 'paper')
from log import log_row  # noqa: E402

OUT = Path('experiments/honest/out')
OUT.mkdir(parents=True, exist_ok=True)
GCFG = dict(red='wpca', d=128, se_shrink='lw', sp_shrink='lw', lam_floor=1e-4, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0)
W0664 = {'q14v5': 3.0, 'q8v5': 2.0, 'll8v5': 1.0, 'tfidf': 0.25, 'distort': 0.10, 'svc4': 12.0}
GAUSS = {'q14v5': 0.70, 'q8v5': 0.44}
TWIN, ADAPT_W = 0.35, 1.0
NDRAW = 12

cpu = np.load('research/honest_oof/cpu_members_oof.npz', allow_pickle=True)
y, fold, ids = cpu['y'], cpu['fold'], cpu['ids'].astype(str)
train = pd.read_csv('data/train.csv')
assert (train.id.astype(str).to_numpy() == ids).all()
pid = train.prompt_id.to_numpy()
G = {}
for f in sorted(glob.glob('research/honest_oof/gpu/honest_oof_*.npz')):
    z = np.load(f)
    assert (z['ids'].astype(str) == ids).all() and (z['fold'] == fold).all()
    for k in z.files:
        G[k] = z[k]
LORA = [m for m in ('q14v5', 'q8v5', 'll8v5') if f'{m}_plain' in G and bool(G[f'{m}_split_ok'])]
print('LoRA members with honest OOF:', LORA)
for m in LORA:
    print(f'  {m} plain bacc {bas(y, G[m + "_plain"].argmax(1)):.4f}')

# Gemma OOF on our GroupKFold folds 0-2
ofold = np.full(len(y), -1)
for k, (_, va) in enumerate(GroupKFold(5).split(np.zeros(len(y)), y, pid)):
    ofold[va] = k
GEM = {'gemma': np.full((len(y), 18), np.nan), 'gemma_cc': np.full((len(y), 18), np.nan)}
labels = list(cpu['labels'].astype(str))
for f in sorted(glob.glob('research/kernels_2026-10-26/gemma_s2/ts/lora_twostage/fold*/val_outputs.npz')):
    v = np.load(f, allow_pickle=True)   # our own kernel output
    assert list(v['classes']) == labels
    GEM['gemma'][v['rows']] = log_softmax(v['plain_logits'].astype(float), 1)
    GEM['gemma_cc'][v['rows']] = log_softmax(v['center_logits_chunk'].astype(float), 1)
grow = np.isfinite(GEM['gemma']).all(1)
print(f'Gemma OOF rows {grow.sum()} (our folds {sorted(set(ofold[grow]))})')


def draw_groups(rng, keep):
    """random groups of 2-3 inside each prompt over the kept rows; returns group id per row (-1 = not kept)."""
    g = np.full(len(y), -1)
    nxt = 0
    for p in np.unique(pid[keep]):
        ii = rng.permutation(np.flatnonzero(keep & (pid == p)))
        i = 0
        while i < len(ii):
            n = 2 if len(ii) - i in (2, 4) else int(rng.integers(2, 4))
            n = min(n, len(ii) - i)
            g[ii[i:i + n]] = nxt
            nxt += 1
            i += n
    return g


def members(g):
    """member log-probs (unweighted) for the rows with g >= 0, following infer.py."""
    rows = np.flatnonzero(g >= 0)
    gs = g[rows]
    size = np.bincount(gs)[gs]
    M = {}
    for m in LORA:
        M[m] = log_softmax(G[m + '_plain'][rows].astype(float), 1)
        rc = G[m + '_rc'][rows].astype(float)
        mean = np.zeros((gs.max() + 1, 18))
        np.add.at(mean, gs, rc)
        mean /= np.bincount(gs)[:, None]
        cent = log_softmax(rc - mean[gs] + G[m + '_bc'][rows], 1)
        M[m + '_twin'] = np.where((size >= 2)[:, None], cent, 0.0)
        if m in GAUSS:
            out = np.zeros((len(rows), 18))
            for k in range(5):
                prm = {'m': G[m + '_m'][k], 'lam': G[m + '_lam'][k], 'T1': 3.0, 'Tg': 3.0, 'cfg': GCFG}
                sel = (fold[rows] == k) & (size >= 2)
                if sel.any():
                    out[sel] = ga.scores(None, np.unique(gs[sel], return_inverse=True)[1], prm, U=G[m + '_U'][rows[sel]].astype(float))
            M[m + '_gauss'] = out
    for m in ('tfidf', 'distort', 'svc4'):
        M[m] = cpu[m][rows].astype(float)
    for m in GEM:
        M[m] = np.nan_to_num(GEM[m][rows])
    return rows, M


def weights_0664():
    w = {}
    for m in LORA:
        w[m] = W0664[m]
        w[m + '_twin'] = ADAPT_W * W0664[m] * TWIN
        if m in GAUSS:
            w[m + '_gauss'] = GAUSS[m]
    for m in ('tfidf', 'distort', 'svc4'):
        w[m] = W0664[m]
    return w


def blend(M, w, sinkhorn=0.0, mix=0.35):
    tot = sum(w[k] * M[k] for k in w if w[k] > 0)
    n = sum(1 for k in w if w[k] > 0)
    if sinkhorn <= 0:
        return tot
    P = softmax(tot / n, 1)
    prior = np.bincount(y, minlength=18) + 1.0
    prior /= prior.sum()
    tgt = (1 - mix) * P.mean(0) + mix * prior
    tgt /= tgt.sum()
    Q = P.copy()
    for _ in range(100):
        Q *= len(Q) * tgt[None] / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
        Q /= Q.sum(1, keepdims=True)
    return (1 - sinkhorn) * np.log(P + 1e-12) + sinkhorn * np.log(Q + 1e-12)


def fit_nnls(Ms, ys, names):
    A = np.concatenate([np.stack([softmax(M[k], 1).reshape(-1) for k in names], 1) for M in Ms])
    b = np.concatenate([np.eye(18)[yy].reshape(-1) for yy in ys])
    return dict(zip(names, nnls(A, b)[0]))


def scale_to(w, ref):
    s = sum(ref.values()) / max(sum(w.values()), 1e-12)
    return {k: v * s for k, v in w.items()}


base_w = weights_0664()
names = list(base_w)
rng = np.random.default_rng(0)
draws = []
for d in range(NDRAW):
    g = draw_groups(rng, np.ones(len(y), bool))
    draws.append((g, *members(g)))
print(f'{NDRAW} draws, groups of 2-3, rows per draw {len(draws[0][1])}')

# 1) 0.664 weights vs NNLS (fit on 4 seeded folds, scored on the 5th), no Sinkhorn
res = []
pred_b, pred_n, W_n = [], [], []
for g, rows, M in draws:
    yb = y[rows]
    pb = blend(M, base_w).argmax(1)
    pn = np.empty(len(rows), int)
    for k in range(5):
        tr, te = fold[rows] != k, fold[rows] == k
        w = fit_nnls([{n: M[n][tr] for n in names}], [yb[tr]], names)
        W_n.append(w)
        pn[te] = blend({n: M[n][te] for n in names}, w).argmax(1)
    res.append(dict(base=bas(yb, pb), nnls=bas(yb, pn)))
R = pd.DataFrame(res)
wn = pd.DataFrame(W_n).mean()
print(f"\n0.664 weights {R.base.mean():.4f} +- {R.base.std():.4f} | NNLS {R.nnls.mean():.4f} +- {R.nnls.std():.4f} | "
      f"diff {(R.nnls - R.base).mean():+.4f} (better in {(R.nnls > R.base).sum()}/{len(R)} draws)")
cmp = pd.DataFrame({'w_0664': pd.Series(base_w), 'w_nnls_scaled': pd.Series(scale_to(wn.to_dict(), base_w))})
cmp['share_0664'] = cmp.w_0664 / cmp.w_0664.sum()
cmp['share_nnls'] = cmp.w_nnls_scaled / cmp.w_nnls_scaled.sum()
print(cmp.round(3).to_string())

# 2) leave one member out (0.664 weights)
loo = {}
for k in names:
    w = dict(base_w, **{k: 0.0})
    loo[k] = np.mean([bas(y[rows], blend(M, w).argmax(1)) for g, rows, M in draws]) - R.base.mean()
print('\nleave-one-out change (0.664 weights):', {k: round(v, 4) for k, v in sorted(loo.items(), key=lambda t: t[1])})

# 3) Sinkhorn strength under class-mix shift (0.664 weights and NNLS weights)
wn_s = scale_to(wn.to_dict(), base_w)
sk = []
for alpha in (None, 3.0, 1.0):
    for d, (g0, rows0, M0) in enumerate(draws[:6]):
        r2 = np.random.default_rng(100 + d)
        if alpha is None:
            keep = np.ones(len(rows0), bool)
        else:
            cw = r2.dirichlet(np.full(18, alpha))
            keep = r2.random(len(rows0)) < (cw / cw.max())[y[rows0]]
        Mk = {k: v[keep] for k, v in M0.items()}
        yk = y[rows0][keep]
        for s in (0.0, 0.3, 0.45, 0.6, 1.0):
            sk.append(dict(alpha='uniform' if alpha is None else alpha, draw=d, s=s,
                           base=bas(yk, blend(Mk, base_w, s).argmax(1)), nnls=bas(yk, blend(Mk, wn_s, s).argmax(1))))
SK = pd.DataFrame(sk).groupby(['alpha', 's'])[['base', 'nnls']].mean().unstack('alpha')
print('\nSinkhorn strength (rows: s; cols: class mix; mean over 6 draws):')
print(SK.round(4).to_string())

# 4) Gemma as an extra member (rows of our folds 0-2), NNLS with vs without
gres, gw = [], []
for g, rows, M in draws:
    sel = grow[rows]
    Ms = {k: v[sel] for k, v in M.items()}
    ys, fs = y[rows][sel], fold[rows][sel]
    p0, p1 = np.empty(sel.sum(), int), np.empty(sel.sum(), int)
    for k in range(5):
        tr, te = fs != k, fs == k
        if not te.any():
            continue
        w0 = fit_nnls([{n: Ms[n][tr] for n in names}], [ys[tr]], names)
        w1 = fit_nnls([{n: Ms[n][tr] for n in names + ['gemma', 'gemma_cc']}], [ys[tr]], names + ['gemma', 'gemma_cc'])
        gw.append(w1)
        p0[te] = blend({n: Ms[n][te] for n in names}, w0).argmax(1)
        p1[te] = blend({n: Ms[n][te] for n in names + ['gemma', 'gemma_cc']}, w1).argmax(1)
    gres.append(dict(without=bas(ys, p0), with_gemma=bas(ys, p1)))
GR = pd.DataFrame(gres)
gws = pd.DataFrame(gw).mean()
gshare = (gws['gemma'] + gws['gemma_cc']) / gws.sum()
print(f"\nGemma (rows {grow.sum()}): NNLS without {GR.without.mean():.4f} | with Gemma {GR.with_gemma.mean():.4f} | "
      f"diff {(GR.with_gemma - GR.without).mean():+.4f} (better in {(GR.with_gemma > GR.without).sum()}/{len(GR)}) | "
      f"Gemma weight share {gshare:.3f} (plain {gws['gemma'] / gws.sum():.3f}, chunk-centred {gws['gemma_cc'] / gws.sum():.3f})")

R.to_csv(OUT / 'stack_sim_base_vs_nnls.csv', index=False)
cmp.to_csv(OUT / 'stack_sim_weights.csv')
SK.to_csv(OUT / 'stack_sim_sinkhorn.csv')
GR.to_csv(OUT / 'stack_sim_gemma.csv', index=False)
json.dump({'members': LORA, 'loo': loo, 'nnls_weights_scaled': wn_s, 'gemma_share': gshare}, open(OUT / 'stack_sim.json', 'w'), indent=1)
log_row(id='E066' + ('' if 'q14v5' in LORA else 'a'), category='honest-oof',
        metric='bacc_sim_groups2-3_seeded_5fold', description='honest test-like sim of the 0.664 blend subset (' + '+'.join(LORA) + ' + tfidf/distort/svc4): 0.664 weights vs NNLS, LOO, Sinkhorn under class shift, Gemma as extra member',
        config=f'{NDRAW} draws of 2-3 passage groups per prompt; NNLS fit on 4 seeded folds; Gemma on our folds 0-2', baseline=f'{R.base.mean():.4f}',
        value=f'{R.nnls.mean():.4f}', folds_pos=f'{(R.nnls > R.base).sum()}/{len(R)} draws',
        files='experiments/honest/out/stack_sim_*.csv',
        notes=f"Gemma diff {(GR.with_gemma - GR.without).mean():+.4f} share {gshare:.3f}; best Sinkhorn s (uniform) base "
              f"{SK['base']['uniform'].idxmax()}")

# 5) Gemma on top of the 0.664 hand weights: weights (plain a, chunk-centred b) chosen on 4 seeded folds by grid, scored
#    on the 5th (rows of our folds 0-2); plus sanity: each Gaussian head alone on grouped rows
GRID = [(a, b) for a in (0, 0.5, 1, 2) for b in (0, 0.5, 1, 2, 3, 4)]
g5, picks = [], []
for g, rows, M in draws:
    sel = grow[rows]
    Ms = {k: v[sel] for k, v in M.items()}
    ys, fs = y[rows][sel], fold[rows][sel]
    p0, p1 = blend(Ms, base_w).argmax(1), np.empty(sel.sum(), int)
    for k in range(5):
        tr, te = fs != k, fs == k
        if not te.any():
            continue
        sc = {ab: bas(ys[tr], blend({n: v[tr] for n, v in Ms.items()}, dict(base_w, gemma=ab[0], gemma_cc=ab[1])).argmax(1)) for ab in GRID}
        ab = max(sc, key=sc.get)
        picks.append(ab)
        p1[te] = blend({n: v[te] for n, v in Ms.items()}, dict(base_w, gemma=ab[0], gemma_cc=ab[1])).argmax(1)
    g5.append(dict(base=bas(ys, p0), base_plus_gemma=bas(ys, p1)))
G5 = pd.DataFrame(g5)
print(f"\nGemma on 0.664 hand weights (rows {grow.sum()}): base {G5.base.mean():.4f} | + Gemma {G5.base_plus_gemma.mean():.4f} | "
      f"diff {(G5.base_plus_gemma - G5.base).mean():+.4f} (better in {(G5.base_plus_gemma > G5.base).sum()}/{len(G5)}) | "
      f"picked (plain, centred) weights: {pd.Series(picks).value_counts().head(4).to_dict()}")
for m in [m for m in LORA if m in GAUSS]:
    acc = [bas(y[rows][M[m + '_gauss'].any(1)], M[m + '_gauss'][M[m + '_gauss'].any(1)].argmax(1)) for g, rows, M in draws[:3]]
    print(f'{m}_gauss alone on grouped rows: {np.mean(acc):.4f} (plain {bas(y, G[m + "_plain"].argmax(1)):.4f})')
G5.to_csv(OUT / 'stack_sim_gemma_on_0664.csv', index=False)
