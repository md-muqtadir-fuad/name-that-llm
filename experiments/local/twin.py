"""Plain + 'twin' (prompt-chunk-centred) LinearSVC on TF-IDF + stylometry, 5-fold prompt-grouped.

Because both heads are linear, applying the twin to group-centred features equals
    twin_raw(x_i) - mean_{j in group} twin_raw(x_j) + b,
so we store OOF raw scores (without intercept) and centre them later for any grouping.
Saves out/twin_scores.npz with: S (plain decision), T (twin raw, no intercept), b (twin intercept per row's fold),
Tp/Tt (temperatures fitted on OOF NLL).
"""
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import OUT, K, load_train, folds, bacc, log_softmax, stylo_features

df = load_train()
y, g = df.y.values, df.g.values
fold = folds(df)
texts = df.text.values
STY = stylo_features(texts).drop(columns=["n_chars"]).values.astype(float)
ANALYZER = sys.argv[1] if len(sys.argv) > 1 else "char_wb"
LOWER = ANALYZER == "char_wb"
STY_W = 0.15
C = 0.5


def random_chunks(groups, size, rng):
    lab = np.empty(len(groups), int)
    nxt = 0
    for p in np.unique(groups):
        m = rng.permutation(np.flatnonzero(groups == p))
        nb = max(1, int(round(len(m) / size)))
        parts = np.array_split(m, nb)
        for part in parts:
            lab[part] = nxt
            nxt += 1
    return lab


def center_sparse(X, lab):
    """X - groupmean(X) for sparse X and group labels."""
    n = X.shape[0]
    u, inv = np.unique(lab, return_inverse=True)
    cnt = np.bincount(inv)
    G = sparse.csr_matrix((1.0 / cnt[inv], (inv, np.arange(n))), shape=(len(u), n))
    M = G @ X  # group means
    B = sparse.csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, len(u)))
    return (X - B @ M).tocsr()


def run_fold(k):
    t0 = time.time()
    if k == "final":  # fit on all train, score the external distractor passages
        tr = np.ones(len(df), bool)
        va_texts = EXT_TEXTS
    else:
        tr, va = fold != k, fold == k
    vecs = [
        TfidfVectorizer(analyzer=ANALYZER, ngram_range=(1, 5), min_df=2, sublinear_tf=True, lowercase=LOWER,
                        max_features=300000, dtype=np.float32),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=LOWER,
                        token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=200000, dtype=np.float32),
    ]
    sc = StandardScaler().fit(STY[tr])
    def feats(tx, sty, fit=False):
        parts = [(v.fit_transform(tx) if fit else v.transform(tx)) for v in vecs]
        parts.append(sparse.csr_matrix(np.clip(sc.transform(sty), -5, 5) * STY_W))
        return sparse.hstack(parts).tocsr().astype(np.float32)
    Xtr = feats(texts[tr], STY[tr], fit=True)
    if k == "final":
        Xva = feats(EXT_TEXTS, EXT_STY)
    else:
        Xva = feats(texts[va], STY[va])
    plain = LinearSVC(C=C).fit(Xtr, y[tr])
    S = plain.decision_function(Xva)
    rng = np.random.default_rng(1000 + (k if k != "final" else 99))
    gtr = g[tr]
    Xc = sparse.vstack([center_sparse(Xtr, random_chunks(gtr, s, rng)) for s in (2, 3, 4, 6, 18)]).tocsr()
    yc = np.tile(y[tr], 5)
    twin = LinearSVC(C=C).fit(Xc, yc)
    T = Xva @ twin.coef_.T  # raw, no intercept
    if k == "final":
        print(f"final fit {time.time()-t0:.0f}s", flush=True)
        return k, None, S, np.asarray(T), twin.intercept_
    print(f"fold {k}: plain {bacc(y[va], S.argmax(1)):.4f}, Xc nnz {Xc.nnz/1e6:.1f}M, {time.time()-t0:.0f}s", flush=True)
    return k, np.flatnonzero(va), S, np.asarray(T), twin.intercept_


import pandas as pd
from common import ROOT
EXT = pd.read_csv(ROOT / "research/ext_data/extra_api/ntl_extra_passages_claude.csv")
EXT_TEXTS = EXT.text.values
EXT_STY = stylo_features(EXT_TEXTS).drop(columns=["n_chars"]).values.astype(float)
res = Parallel(n_jobs=3)(delayed(run_fold)(k) for k in [0, 1, 2, 3, 4, "final"])
S = np.zeros((len(df), K)); T = np.zeros((len(df), K)); B = np.zeros((len(df), K))
for k, idx, s, t, b in res:
    if k == "final":
        S_ext, T_ext, B_ext = s, t, np.tile(b, (len(s), 1))
    else:
        S[idx], T[idx], B[idx] = s, t, b
fS = lambda tt: -log_softmax(S * tt)[np.arange(len(y)), y].mean()
Tp = minimize_scalar(fS, bounds=(0.1, 50), method="bounded").x
# twin evaluated with true full-prompt centring (oracle groups) for a quick sanity check
Tc = T.copy()
for p in np.unique(g):
    m = g == p
    Tc[m] = T[m] - T[m].mean(0)
Tc += B
fT = lambda tt: -log_softmax(Tc * tt)[np.arange(len(y)), y].mean()
Tt = minimize_scalar(fT, bounds=(0.1, 50), method="bounded").x
print(f"plain OOF bacc {bacc(y, S.argmax(1)):.4f}  plain score-centred (true prompts) "
      f"{bacc(y, (S - np.vstack([S[g==p].mean(0) for p in g])).argmax(1)):.4f}  "
      f"twin centred (true prompts) {bacc(y, Tc.argmax(1)):.4f}  Tp {Tp:.2f} Tt {Tt:.2f}")
np.savez(OUT / f"twin_scores_{ANALYZER}.npz", S=S, T=T, B=B, Tp=Tp, Tt=Tt, fold=fold,
         S_ext=S_ext, T_ext=T_ext, B_ext=B_ext, ext_g=EXT.prompt_id.astype("category").cat.codes.values + 10000)
