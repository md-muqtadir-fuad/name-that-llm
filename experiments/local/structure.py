"""Prompt-structure tools: pool simulation, prompt-group recovery, group/global decoders."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

from common import K, log_softmax, softmax

# ------------------------------------------------------------------ scenarios
SCENARIOS = {
    # name: (description, sampler)
    "full18": "all ~18 answers per prompt, balanced classes",
    "drop0-30": "drop U(0,30%) of each prompt's answers at random, classes balanced in expectation",
    "uneq_keep": "per-class keep rate U(0.35,1) (unequal class counts), ~12 per prompt",
    "n3-8": "3-8 answers per prompt, unequal class weights",
    "n1-5": "1-5 answers per prompt, unequal class weights",
    "n2-3": "2-3 answers per prompt, unequal class weights",
    "n1-2": "1-2 answers per prompt, unequal class weights",
}


def sample_pool(g, y, scenario, rng):
    """Return sorted row indices of a simulated test pool. g = prompt ids (int), y = labels."""
    keep = []
    cw = rng.uniform(0.35, 1.0, size=K)  # class weights -> unequal class counts
    for p in np.unique(g):
        rows = np.flatnonzero(g == p)
        if scenario == "full18":
            keep.extend(rows)
        elif scenario == "drop0-30":
            f = rng.uniform(0, 0.3)
            n = len(rows) - int(round(f * len(rows)))
            keep.extend(rng.choice(rows, n, replace=False))
        elif scenario == "uneq_keep":
            keep.extend(rows[rng.random(len(rows)) < cw[y[rows]]])
        else:
            lo, hi = {"n3-8": (3, 8), "n1-5": (1, 5), "n2-3": (2, 3), "n1-2": (1, 2)}[scenario]
            n = rng.integers(lo, hi + 1)
            w = cw[y[rows]]
            keep.extend(rng.choice(rows, min(n, len(rows)), replace=False, p=w / w.sum()))
    return np.sort(np.array(keep, dtype=int))


# ------------------------------------------------------------ group recovery
def content_tfidf(texts, ngram=(1, 1), max_df=0.5):
    v = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=ngram, min_df=1,
                        max_df=max_df, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b", dtype=np.float32)
    X = v.fit_transform(texts)
    return X


def cosine_dist(X):
    S = (X @ X.T).toarray() if hasattr(X, "toarray") else X @ X.T
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0, 2)


def agglo(D, thr, method="average"):
    if len(D) == 1:
        return np.zeros(1, int)
    Z = linkage(squareform(D, checks=False), method=method)
    return fcluster(Z, thr, criterion="distance") - 1


def agglo_multi(D, thrs, method="average"):
    """One linkage, several distance cuts."""
    if len(D) == 1:
        return {t: np.zeros(1, int) for t in thrs}
    Z = linkage(squareform(D, checks=False), method=method)
    return {t: fcluster(Z, t, criterion="distance") - 1 for t in thrs}


def cap_clusters(labels, D, cap=18):
    """Split clusters larger than cap by re-running average linkage into ceil(n/cap) parts."""
    labels = labels.copy()
    nxt = labels.max() + 1
    for c in np.unique(labels):
        m = np.flatnonzero(labels == c)
        if len(m) > cap:
            Z = linkage(squareform(D[np.ix_(m, m)], checks=False), method="average")
            sub = fcluster(Z, int(np.ceil(len(m) / cap)), criterion="maxclust")
            for s in np.unique(sub)[1:]:
                labels[m[sub == s]] = nxt
                nxt += 1
    return labels


def cluster_metrics(true, pred, y=None):
    ari = adjusted_rand_score(true, pred)
    h, c, v = homogeneity_completeness_v_measure(true, pred)
    # pair precision / recall
    n = len(true)
    same_t = true[:, None] == true[None, :]
    same_p = pred[:, None] == pred[None, :]
    iu = np.triu_indices(n, 1)
    tp = (same_t & same_p)[iu].sum()
    pp = same_p[iu].sum()
    tt = same_t[iu].sum()
    sizes = np.bincount(pred)[pred]
    out = dict(ari=ari, homog=h, compl=c, pair_prec=tp / max(pp, 1), pair_rec=tp / max(tt, 1),
               singleton_frac=float(np.mean(sizes == 1)), n_clusters=len(np.unique(pred)),
               mean_size=float(np.mean(sizes)))
    # purity: fraction of rows whose cluster majority prompt == own prompt
    pur = 0
    for c in np.unique(pred):
        m = pred == c
        pur += np.bincount(true[m]).max()
    out["purity"] = pur / n
    if y is not None:  # fraction of clusters (size>=2) that violate one-model-per-group
        viol = tot = 0
        for c in np.unique(pred):
            m = np.flatnonzero(pred == c)
            if len(m) >= 2:
                tot += 1
                viol += len(np.unique(y[m])) < len(m)
        out["dup_label_clusters"] = viol / max(tot, 1)
    return out


# --------------------------------------------------------------- decoders
def groups_iter(groups):
    order = np.argsort(groups, kind="stable")
    gs = groups[order]
    cuts = np.flatnonzero(np.diff(gs)) + 1
    return np.split(order, cuts)


def center(L, groups, beta=1.0, min_size=2):
    """Subtract beta x group-mean of the scores (column-wise) inside each group (adaptive centering on scores)."""
    Z = L.copy()
    for m in groups_iter(groups):
        if len(m) >= min_size:
            Z[m] = L[m] - beta * L[m].mean(0, keepdims=True)
    return Z


def repulsion(L, groups, alpha=0.2):
    """Public-D logit repulsion: z_i - alpha * mean_{j != i} z_j."""
    Z = L.copy()
    for m in groups_iter(groups):
        n = len(m)
        if n >= 2:
            s = L[m].sum(0, keepdims=True)
            Z[m] = L[m] - alpha * (s - L[m]) / (n - 1)
    return Z


def hungarian(L, groups):
    """Hard one-passage-per-model assignment inside each group (max sum log p)."""
    pred = L.argmax(1).copy()
    for m in groups_iter(groups):
        n = len(m)
        if n < 2:
            continue
        reps = int(np.ceil(n / K))
        cost = -np.tile(L[m], (1, reps))
        r, c = linear_sum_assignment(cost)
        pred[m[r]] = c % K
    return pred


def soft_distinct(L, groups, tau=1.0, iters=50):
    """Approximate P(author_i = c | all authors in group distinct) by Sinkhorn on an augmented
    (n + 1) x K plan: n passage rows of mass 1, one slack row of mass K - n, K columns of mass 1.
    Returns log-marginals (rows renormalised)."""
    Z = L.copy()
    for m in groups_iter(groups):
        n = len(m)
        if n < 2 or n > K:
            continue
        P = softmax(L[m] / tau)
        A = np.vstack([P, np.full((1, K), 1.0 / K)])
        rmass = np.r_[np.ones(n), K - n]
        for _ in range(iters):
            A *= (1.0 / np.maximum(A.sum(0, keepdims=True), 1e-300))
            A *= (rmass / np.maximum(A.sum(1), 1e-300))[:, None]
        Q = A[:n] / A[:n].sum(1, keepdims=True)
        Z[m] = np.log(np.maximum(Q, 1e-300))
    return Z


def exact_distinct(L, groups, tau=1.0, max_n=8):
    """Exact marginals P(a_i=c | distinct) via DP over passage subsets (n <= max_n); else Sinkhorn."""
    Z = L.copy()
    for m in groups_iter(groups):
        n = len(m)
        if n < 2:
            continue
        if n > max_n:
            Z[m] = soft_distinct(L[m], np.zeros(n, int), tau)
            continue
        P = softmax(L[m] / tau)  # n x K
        S = 1 << n
        # forward: f[k][mask] = sum over assignments of passages in mask to distinct classes among first k classes
        f = np.zeros((K + 1, S))
        f[0, 0] = 1.0
        for k in range(K):
            f[k + 1] = f[k]
            for i in range(n):
                bit = 1 << i
                masks = np.arange(S)
                sel = (masks & bit) != 0
                f[k + 1, sel] += f[k, masks[sel] ^ bit] * P[i, k]
        b = np.zeros((K + 2, S))
        full = S - 1
        b[K, full] = 1.0  # b[k][mask]: ways to complete from class k given mask already used
        for k in range(K - 1, -1, -1):
            b[k] = b[k + 1]
            for i in range(n):
                bit = 1 << i
                masks = np.arange(S)
                sel = (masks & bit) == 0
                b[k, sel] += P[i, k] * b[k + 1, masks[sel] | bit]
        Q = np.zeros((n, K))
        masks = np.arange(S)
        for i in range(n):
            bit = 1 << i
            sel = (masks & bit) == 0
            for k in range(K):
                Q[i, k] = P[i, k] * np.sum(f[k, sel] * b[k + 1, masks[sel] | bit])
        Q /= np.maximum(Q.sum(1, keepdims=True), 1e-300)
        Z[m] = np.log(np.maximum(Q, 1e-300))
    return Z


def sinkhorn_global(L, alpha=1.0, iters=100):
    """Balance predicted class mass over the pool. alpha=1 -> uniform target, 0.5 -> 0.541-notebook partial."""
    P = softmax(L)
    N = len(P)
    target = (1 - alpha) * P.sum(0) + alpha * N / K
    for _ in range(iters):
        P *= (target / np.maximum(P.sum(0), 1e-300))[None, :]
        P /= P.sum(1, keepdims=True)
    return np.log(np.maximum(P, 1e-300))


def bias_balance_counts(L, iters=300, lr=0.5):
    """Adjust per-class biases until argmax counts are ~uniform."""
    b = np.zeros(K)
    N = len(L)
    for _ in range(iters):
        cnt = np.bincount((L + b).argmax(1), minlength=K) + 0.5
        b -= lr * np.log(cnt / (N / K))
        b -= b.mean()
    return L + b
