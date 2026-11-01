"""E069: per-class bias correction of the blend. On honest OOF the members over-predict some classes (plain-head blend:
gemini-3.8-flash 182 vs 3.7 101 of 150/150; tfidf: claude-fable-5 212 vs glm-5.3 44). Test: learn an 18-dim offset b
on 4 seeded folds, apply it to the 5th, then the 0.664 Sinkhorn (0.45, target mix 0.35) on the whole sim test set.
  prior: b = -tau * log(mean predicted prob), tau picked on the training folds
  fit:   b = argmin class-balanced cross-entropy of softmax(lpn + b) on the training folds (+ small L2)
Blend = members we can score honestly at the 0.664 hand weights (lpn = sum(w * lp) / n_members, as infer.py), groups of
2-3 answers per prompt; optionally + our Gemma (plain 2, centred 2.8; rows of our folds 0-2). Class mixes: uniform and
Dirichlet(3) subsamples. Run from the repo root: python experiments/honest/e069_bias.py"""
import glob
import sys

import numpy as np
import pandas as pd
from safetensors.numpy import load_file
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.model_selection import GroupKFold

sys.path.insert(0, 'nb/honest_oof')
import gauss as ga  # noqa: E402

sys.path.insert(0, 'paper')
from log import log_row  # noqa: E402

cpu = np.load('research/honest_oof/cpu_members_oof.npz', allow_pickle=True)
y, fold = cpu['y'], cpu['fold']
train = pd.read_csv('data/train.csv')
pid = train.prompt_id.to_numpy()
G = {}
for f in sorted(glob.glob('research/honest_oof/gpu/honest_oof_*.npz')):
    z = np.load(f)
    G.update({k: z[k] for k in z.files})
GCFG = dict(red='wpca', d=128, se_shrink='lw', sp_shrink='lw', lam_floor=1e-4, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0)
W = {'q14v5': 3.0, 'q8v5': 2.0, 'll8v5': 1.0}
GAUSS = {'q14v5': 0.70, 'q8v5': 0.44}
ofold = np.full(len(y), -1)
for k, (_, va) in enumerate(GroupKFold(5).split(np.zeros(len(y)), y, pid)):
    ofold[va] = k
GP, GRC, GBC = (np.full((len(y), 18), np.nan) for _ in range(3))
for d in sorted(glob.glob('research/kernels_2026-10-26/gemma_s2/ts/lora_twostage/fold[0-9]')):
    v = np.load(f'{d}/val_outputs.npz', allow_pickle=True)   # our own kernel output
    H = load_file(f'{d}/heads.safetensors')
    GP[v['rows']] = v['plain_logits']
    GRC[v['rows']] = v['z'].astype(np.float32) @ H['centered.weight'].astype(np.float32).T
    GBC[v['rows']] = H['centered.bias']
grow = np.isfinite(GP).all(1)


def twin(rc, bc, g):
    mean = np.zeros((g.max() + 1, 18)); np.add.at(mean, g, rc); mean /= np.bincount(g)[:, None]
    return log_softmax(rc - mean[g] + bc, 1)


def blend(g, gemma):
    """lpn = sum(w * lp) / n_members over all rows, groups g (all groups have 2-3 rows)."""
    tot, n = np.zeros((len(y), 18)), 0
    for m, w in W.items():
        tot += w * log_softmax(G[m + '_plain'].astype(float), 1) + 0.35 * w * twin(G[m + '_rc'].astype(float), G[m + '_bc'], g)
        n += 2
        if m in GAUSS:
            out = np.zeros((len(y), 18))
            for k in range(5):
                sel = fold == k
                prm = {'m': G[m + '_m'][k], 'lam': G[m + '_lam'][k], 'T1': 3.0, 'Tg': 3.0, 'cfg': GCFG}
                out[sel] = ga.scores(None, np.unique(g[sel], return_inverse=True)[1], prm, U=G[m + '_U'][sel].astype(float))
            tot += GAUSS[m] * out
            n += 1
    tot += 0.25 * cpu['tfidf'] + 0.10 * cpu['distort'] + 12 * cpu['svc4']
    n += 3
    if gemma:
        tot += 2.0 * log_softmax(np.nan_to_num(GP), 1) + 2.8 * twin(np.nan_to_num(GRC), np.nan_to_num(GBC), g)
        n += 2
    return tot / n


def sinkhorn(lpn, s=0.45, mix=0.35):
    if s <= 0:
        return lpn
    P = softmax(lpn, 1)
    prior = np.bincount(y, minlength=18) + 1.0
    tgt = (1 - mix) * P.mean(0) + mix * prior / prior.sum()
    Q = P.copy()
    for _ in range(100):
        Q *= len(Q) * tgt[None] / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
        Q /= Q.sum(1, keepdims=True)
    return (1 - s) * np.log(P + 1e-12) + s * np.log(Q + 1e-12)


def fit_offset(lpn, yy, l2=0.01):
    cw = 1.0 / np.bincount(yy, minlength=18)[yy]
    def f(b):
        lp = log_softmax(lpn + b, 1)
        p = np.exp(lp)
        loss = -(cw * lp[np.arange(len(yy)), yy]).sum() / cw.sum() + l2 * (b ** 2).sum()
        gb = p * cw[:, None]
        gb[np.arange(len(yy)), yy] -= cw
        return loss, gb.sum(0) / cw.sum() + 2 * l2 * b
    return minimize(f, np.zeros(18), jac=True, method='L-BFGS-B').x


def corrected(lpn, rows, method):
    """per seeded fold: offset learned on the other folds' rows (within `rows`), applied to this fold."""
    out = lpn[rows].copy()
    for k in range(5):
        tr, te = rows[fold[rows] != k], fold[rows] == k
        if method == 'prior':
            best = None
            for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
                b = -tau * np.log(softmax(lpn[tr], 1).mean(0))
                sc = bas(y[tr], sinkhorn(lpn[tr] + b).argmax(1))
                if best is None or sc > best[0]:
                    best = (sc, b)
            b = best[1]
        else:
            b = fit_offset(lpn[tr], y[tr])
        out[te] += b
    return out


rng = np.random.default_rng(0)
res = []
for d in range(8):
    g = np.full(len(y), -1); nxt = 0
    for p_ in np.unique(pid):
        ii = rng.permutation(np.flatnonzero(pid == p_)); i = 0
        while i < len(ii):
            n = 2 if len(ii) - i in (2, 4) else int(rng.integers(2, 4))
            g[ii[i:i + n]] = nxt; nxt += 1; i += n
    for setup, gemma, rows in (('0.664 subset', False, np.arange(len(y))), ('0.664 subset + Gemma', True, np.flatnonzero(grow))):
        lpn = blend(g, gemma)
        V = {'none': lpn[rows], 'prior': corrected(lpn, rows, 'prior'), 'fit': corrected(lpn, rows, 'fit')}
        r2 = np.random.default_rng(1000 + d)
        mixes = [('uniform', np.ones(len(rows), bool))]
        for j in range(3):
            cw = r2.dirichlet(np.full(18, 3.0))
            mixes.append((f'dirichlet3', r2.random(len(rows)) < (cw / cw.max())[y[rows]]))
        for mix, keep in mixes:
            for meth, L in V.items():
                res.append(dict(draw=d, setup=setup, mix=mix, method=meth,
                                bacc=bas(y[rows][keep], sinkhorn(L[keep]).argmax(1)),
                                bacc_no_sk=bas(y[rows][keep], L[keep].argmax(1))))
R = pd.DataFrame(res)
S = R.groupby(['setup', 'mix', 'method'])[['bacc', 'bacc_no_sk']].mean().unstack('method')
print(S.round(4).to_string())
P = R.pivot_table(index=['setup', 'mix', 'draw'], columns='method', values='bacc', aggfunc='mean')
for (setup, mix), q in P.groupby(level=[0, 1]):
    for meth in ('prior', 'fit'):
        dlt = q[meth] - q['none']
        print(f'{setup:22s} {mix:10s} {meth:5s} vs none: {dlt.mean():+.4f} (se {dlt.std() / np.sqrt(len(dlt)):.4f}, better {(dlt > 0).sum()}/{len(dlt)})')
R.to_csv('experiments/honest/out/e069_bias.csv', index=False)
q = P.loc[('0.664 subset + Gemma', 'uniform')]
log_row(id='E069', category='calibration', metric='bacc_sim_seeded_5fold_sinkhorn045',
        description='per-class bias offset of the blend learned on 4 seeded folds of honest OOF (prior-correction tau / class-balanced CE fit), applied before Sinkhorn 0.45',
        config='8 draws of 2-3 groups; 0.664 hand weights; with/without Gemma (plain 2, centred 2.8); uniform + Dirichlet(3) mixes',
        baseline=f"{q['none'].mean():.4f}", value=f"{q['fit'].mean():.4f}", folds_pos=f"{(q['fit'] > q['none']).sum()}/{len(q)} draws",
        files='experiments/honest/out/e069_bias.csv',
        notes='; '.join(f"{s_} {m_}: fit {(P.loc[(s_, m_)]['fit'] - P.loc[(s_, m_)]['none']).mean():+.4f}, prior "
                        f"{(P.loc[(s_, m_)]['prior'] - P.loc[(s_, m_)]['none']).mean():+.4f}"
                        for s_ in ('0.664 subset', '0.664 subset + Gemma') for m_ in ('uniform', 'dirichlet3')))
