"""Offline tests of research-paper ideas on cached frozen Qwen3-8B features (5-fold GroupKFold by prompt).
Base head = the 0.541 notebook's frozen head: StandardScaler + LogisticRegression(C=0.0025) on layers 7+22 mean pools,
with test-set standardisation. Each idea is scored by balanced accuracy on held-out prompts; group ideas are scored on
simulated test pools with 1-3 passages per prompt (the likely hidden-test regime) using TRUE and RECOVERED groups.

Ideas (paper -> test):
  NAP/INLP topic-subspace removal (Wegmann'22 / nuisance attribute projection): project out top-k PCs of train prompt means
  shrinkage centring (Recursion'19 style per-plate normalisation, softened): x - n/(n+k) * group mean
  hierarchical lab->model (DeTeCtive multi-level; DA-MTL): log p(model) + beta * log p(lab(model))
  prototype / nearest-centroid head (DeTeCtive kNN head): cosine to class centroids, blended with LR
  transductive pseudo-labelling (DAIGT 3rd, Recursion 1st): refit with confident held-out rows, soft weight
  layer choice (Idiosyncrasies / FDLLM: middle layers carry style)
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/papers/out'
OUT.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(ROOT / 'data/train.csv')
LABELS = sorted(tr.model.unique())
K = len(LABELS)
y = tr.model.map({l: i for i, l in enumerate(LABELS)}).values
g = tr.prompt_id.values
LAB = {'gpt': 0, 'gem': 1, 'cla': 2, 'glm': 3, 'qwe': 4, 'kim': 5}
lab_of = np.array([LAB[l[:3]] for l in LABELS])
ylab = lab_of[y]

z = np.load(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'experiments/papers/qwen3_8b_train_feats.npz')
assert (z['ids'] == tr.id.values.astype(str)).all()
LAYERS = list(z['layers'])
E = z['E'].astype(np.float32)   # n x L x 4096


def feat(layers):
    return np.concatenate([E[:, LAYERS.index(L)] for L in layers], 1)


folds = list(GroupKFold(5).split(tr, y, g))
T0 = time.time()
log = open(OUT / 'paper_tests.log', 'a', encoding='utf-8')


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + '\n')
    log.flush()


def fit_lr(X, yy, C, w=None):
    return LogisticRegression(C=C, max_iter=3000).fit(X, yy, sample_weight=w)


def oof(X, C=0.0025, nap_k=0, test_std=True, ret_models=False):
    """Plain head OOF log-probs. nap_k > 0: remove top-k PCs of (train-fold) prompt means from all rows first."""
    L = np.zeros((len(y), K))
    models = []
    for a, b in folds:
        Xa, Xb = X[a], X[b]
        if nap_k:
            M = pd.DataFrame(Xa).groupby(g[a]).mean().values
            M = M - M.mean(0)
            _, _, Vt = np.linalg.svd(M, full_matrices=False)
            P = Vt[:nap_k]
            Xa = Xa - (Xa @ P.T) @ P
            Xb = Xb - (Xb @ P.T) @ P
        sc = StandardScaler().fit(Xa)
        clf = fit_lr(sc.transform(Xa), y[a], C)
        Xt = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-6) if test_std else sc.transform(Xb)
        L[b] = clf.predict_log_proba(Xt)
        models.append((sc, clf, Xa, Xb))
    return (L, models) if ret_models else L


def score(L):
    return bacc(y, L.argmax(1))


results = []


def rec(name, val, extra=''):
    results.append((name, val, extra))
    say(f'{name:55s} {val:.4f} {extra}  ({time.time() - T0:.0f}s)')


# ---------------------------------------------------------------- 1) layer choice + base
X722 = feat([7, 22])
L_base, M_base = oof(X722, ret_models=True)
base = score(L_base)
rec('base L7+22 C0.0025 test-std', base)
rec('base without test-std', score(oof(X722, test_std=False)))
for lays in ([4, 12], [12, 16], [7, 12, 22], [4, 7, 12, 16, 22]):
    rec(f'layers {lays}', score(oof(feat(lays))))

# ---------------------------------------------------------------- 2) NAP / INLP topic-subspace removal (works on singletons)
best_nap, L_nap_best = 0, None
for k in (4, 8, 16, 32, 64):
    Ln = oof(X722, nap_k=k)
    s = score(Ln)
    rec(f'NAP remove top-{k} prompt-mean PCs', s, f'delta {s - base:+.4f}')
    if s > (score(L_nap_best) if L_nap_best is not None else -1):
        best_nap, L_nap_best = k, Ln

# ---------------------------------------------------------------- 3) hierarchical lab -> model
Llab = np.zeros((len(y), 6))
for (a, b), (sc, clf, Xa, Xb) in zip(folds, M_base):
    c = fit_lr(sc.transform(Xa), ylab[a], 0.0025)
    Llab[b] = c.predict_log_proba((Xb - Xb.mean(0)) / (Xb.std(0) + 1e-6))
for beta in (0.25, 0.5, 1.0):
    s = score(L_base + beta * Llab[:, lab_of])
    rec(f'hierarchical: + {beta} * log p(lab)', s, f'delta {s - base:+.4f}')

# ---------------------------------------------------------------- 4) prototype (nearest-centroid, cosine) head blend
Lpro = np.zeros((len(y), K))
for (a, b), (sc, clf, Xa, Xb) in zip(folds, M_base):
    A = sc.transform(Xa)
    B = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-6)
    Cn = np.stack([A[y[a] == k].mean(0) for k in range(K)])
    Cn /= np.linalg.norm(Cn, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    Lpro[b] = log_softmax(20 * Bn @ Cn.T, 1)
rec('prototype head alone (cos, T=1/20)', score(Lpro))
for wp in (0.25, 0.5):
    s = score(L_base + wp * Lpro)
    rec(f'LR + {wp} * prototype', s, f'delta {s - base:+.4f}')

# ---------------------------------------------------------------- 5) transductive pseudo-labelling (held-out fold = test)
for frac, wt in ((0.3, 0.5), (0.5, 0.5), (0.5, 1.0)):
    Lpl = np.zeros((len(y), K))
    for (a, b), (sc, clf, Xa, Xb) in zip(folds, M_base):
        Lb = L_base[b]
        srt = np.sort(Lb, 1)
        marg = srt[:, -1] - srt[:, -2]
        pick = np.argsort(-marg)[:int(frac * len(b))]
        Xt = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-6)
        X2 = np.vstack([sc.transform(Xa), Xt[pick]])
        y2 = np.concatenate([y[a], Lb[pick].argmax(1)])
        w2 = np.concatenate([np.ones(len(a)), np.full(len(pick), wt)])
        Lpl[b] = fit_lr(X2, y2, 0.0025, w2).predict_log_proba(Xt)
    s = score(Lpl)
    acc_pl = None
    rec(f'pseudo-label top {int(frac * 100)}% weight {wt}', s, f'delta {s - base:+.4f}')


# ---------------------------------------------------------------- 6) group ideas on simulated pools (1-3 per prompt)
def pools(rng):
    """Per held-out fold: keep 1-3 random passages per prompt. Returns list of (fold_idx, rows)."""
    out = []
    for fi, (a, b) in enumerate(folds):
        rows = []
        for p in np.unique(g[b]):
            idx = b[g[b] == p]
            rows.extend(rng.choice(idx, rng.randint(1, 4), replace=False))
        out.append((fi, np.array(rows)))
    return out


def recovered_groups(rows, thr=0.6):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(tr.text.values[rows])
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform(D, checks=False), 'average'), thr, criterion='distance')


def twin_models():
    """Centred twin per fold: LR on train rows centred by random same-prompt chunks of 2-3 (matches small test groups)."""
    rng = np.random.RandomState(0)
    tw = []
    for a, b in folds:
        Xa = X722[a]
        Xc = np.empty_like(Xa)
        for p in np.unique(g[a]):
            idx = np.where(g[a] == p)[0]
            rng.shuffle(idx)
            for c0 in range(0, len(idx), 3):
                ch = idx[c0:c0 + 3] if len(idx) - c0 > 3 else idx[c0:]
                Xc[ch] = Xa[ch] - Xa[ch].mean(0)
        sc = StandardScaler().fit(Xc)
        tw.append((sc, fit_lr(sc.transform(Xc), y[a], 0.0025)))
    return tw


TW = twin_models()
say('twin heads ready', f'({time.time() - T0:.0f}s)')
agg = {}
for seed in range(4):
    rng = np.random.RandomState(100 + seed)
    for fi, rows in pools(rng):
        a, b = folds[fi]
        sc, clf, Xa, _ = M_base[fi]
        Xp = X722[rows]
        Xt = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-6)
        Lp = clf.predict_log_proba(Xt)
        for gname, G in (('true', g[rows]), ('rec', recovered_groups(rows))):
            sz = pd.Series(G).map(pd.Series(G).value_counts()).values
            grp = sz >= 2
            variants = {'plain': Lp}
            # shrinkage centring of features, plain head (no twin): x - n/(n+k) * mean
            for kk in (1.0, 3.0):
                Xs = Xp.copy()
                for gg in np.unique(G[grp]):
                    i = G == gg
                    Xs[i] -= (i.sum() / (i.sum() + kk)) * Xp[i].mean(0)
                variants[f'shrink-centre k={kk} (plain head)'] = clf.predict_log_proba((Xs - Xs.mean(0)) / (Xs.std(0) + 1e-6))
            # twin head on group-centred features, ADDED to plain (0.541 style) with full vs shrinkage centring
            tsc, tclf = TW[fi]
            for kk, nm in ((0.0, 'full'), (1.0, 'shrink k=1')):
                Xc = np.zeros_like(Xp)
                for gg in np.unique(G[grp]):
                    i = G == gg
                    Xc[i] = Xp[i] - (i.sum() / (i.sum() + kk)) * Xp[i].mean(0)
                Lt = tclf.predict_log_proba(tsc.transform(Xc))
                for tw in (0.5, 1.0):
                    V = Lp.copy()
                    V[grp] += tw * Lt[grp]
                    variants[f'twin(chunks2-3) {nm} add w={tw}'] = V
            for name, V in variants.items():
                agg.setdefault((gname, name), []).append((y[rows], V.argmax(1)))
say(f'pool simulation done ({time.time() - T0:.0f}s)')
for (gname, name), lst in sorted(agg.items()):
    s = np.mean([bacc(t, p) for t, p in lst])
    sp = np.mean([bacc(t, p) for t, p in agg[(gname, 'plain')]])
    rec(f'[pools 1-3/prompt, {gname} groups] {name}', s, f'delta {s - sp:+.4f}')

pd.DataFrame(results, columns=['idea', 'bacc', 'note']).to_csv(OUT / 'paper_tests.csv', index=False)
say('done')
