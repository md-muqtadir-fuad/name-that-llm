"""Similarity sources and clustering algorithms for recovering same-prompt groups in a test pool.
Every similarity function returns a dense float32 n x n matrix S (cosine-like, higher = more similar, diag = 1)."""
import re

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer


# ------------------------------------------------------------------------------------------ similarity sources
def cos_rows(X, center=True):
    X = np.asarray(X, dtype=np.float64)
    if center:
        X = X - X.mean(0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = (X @ X.T).astype(np.float32)
    np.fill_diagonal(S, 1.0)
    return S


def tfidf_sim(texts, **kw):
    """Content-word TF-IDF cosine (the v1 / 0.541 notebook's grouping signal when kw is empty)."""
    opts = dict(stop_words='english', sublinear_tf=True)
    opts.update(kw)
    X = TfidfVectorizer(**opts).fit_transform(texts)
    S = (X @ X.T).toarray().astype(np.float32)
    np.fill_diagonal(S, 1.0)
    return S


_STEM = re.compile(r'(ing|ed|es|s|ly|ment|ness|ation|ations|ions|ion|al|ity|ies)$')


def _stem_tokens(doc):
    toks = re.findall(r"[a-z][a-z\-]+", doc.lower())
    return [_STEM.sub('', t) or t for t in toks if t not in ENGLISH_STOP_WORDS and len(t) > 2]


def tfidf_stem_sim(texts, ngram=(1, 1)):
    X = TfidfVectorizer(tokenizer=_stem_tokens, lowercase=False, token_pattern=None, sublinear_tf=True,
                        ngram_range=ngram).fit_transform(texts)
    S = (X @ X.T).toarray().astype(np.float32)
    np.fill_diagonal(S, 1.0)
    return S


def bm25_sim(texts, k1=1.2, b=0.75):
    """Symmetrised, self-normalised BM25: s_ij = (bm(i->j) + bm(j->i)) / 2 / sqrt(bm(i->i) bm(j->j))."""
    cv = CountVectorizer(tokenizer=_stem_tokens, lowercase=False, token_pattern=None)
    tf = cv.fit_transform(texts).tocsr().astype(np.float64)
    n = tf.shape[0]
    df = np.asarray((tf > 0).sum(0)).ravel()
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
    dl = np.asarray(tf.sum(1)).ravel()
    avg = dl.mean()
    tfc = tf.tocoo()
    w = tfc.data * (k1 + 1) / (tfc.data + k1 * (1 - b + b * dl[tfc.row] / avg))
    Dw = csr_matrix((w, (tfc.row, tfc.col)), shape=tf.shape)
    Q = (tf > 0).astype(np.float64).multiply(idf[None, :]).tocsr()
    B = (Q @ Dw.T).toarray()
    d = np.sqrt(np.clip(np.diag(B), 1e-9, None))
    S = ((B + B.T) / 2 / d[:, None] / d[None, :]).astype(np.float32)
    np.fill_diagonal(S, 1.0)
    return np.clip(S, 0, 1)


def jaccard_sim(texts):
    cv = CountVectorizer(tokenizer=_stem_tokens, lowercase=False, token_pattern=None, binary=True)
    X = cv.fit_transform(texts).astype(np.float32)
    inter = (X @ X.T).toarray()
    sz = np.asarray(X.sum(1)).ravel()
    S = (inter / (sz[:, None] + sz[None, :] - inter + 1e-9)).astype(np.float32)
    np.fill_diagonal(S, 1.0)
    return S


def csls(S, k=10):
    """Cross-domain similarity local scaling: 2 s_ij - r_i - r_j (r = mean of the k largest off-diagonal sims)."""
    T = S.copy()
    np.fill_diagonal(T, -np.inf)
    r = -np.sort(-T, axis=1)[:, :k].mean(1)
    C = 2 * S - r[:, None] - r[None, :]
    np.fill_diagonal(C, C.max())
    return C.astype(np.float32)


def rank_sim(S):
    """Symmetrised within-row percentile rank of each similarity (1 = nearest neighbour)."""
    n = len(S)
    T = S.copy()
    np.fill_diagonal(T, -np.inf)
    order = np.argsort(-T, axis=1)
    R = np.empty_like(T)
    rows = np.arange(n)[:, None]
    R[rows, order] = np.arange(n)[None, :]
    P = 1 - R / (n - 1)
    P = (P + P.T) / 2
    np.fill_diagonal(P, 1.0)
    return P.astype(np.float32)


def zsim(S):
    """Global standardisation of off-diagonal similarities so that sources with different scales can be averaged."""
    iu = np.triu_indices(len(S), 1)
    v = S[iu]
    mu, sd = v.mean(), v.std() + 1e-9
    Z = (S - mu) / sd
    return Z.astype(np.float32)


# ------------------------------------------------------------------------------------------ clustering
def to_dist(S):
    D = 1 - S
    D = np.clip(D, 0, None)
    np.fill_diagonal(D, 0)
    return D


def link(S, method='average'):
    D = to_dist(S).astype(np.float64)
    D = (D + D.T) / 2
    return linkage(squareform(D, checks=False), method=method)


def cut(Z, thr):
    """thr is a similarity threshold: merge while 1 - linkage distance >= thr."""
    return fcluster(Z, 1 - thr, criterion='distance')


def cap_groups(lab, cap):
    """Groups larger than cap become singletons (D notebook) - used for single-linkage / graph methods."""
    lab = np.asarray(lab).copy()
    sz = np.bincount(lab)
    big = sz[lab] > cap
    lab[big] = lab.max() + 1 + np.arange(big.sum())
    return lab


def cc_threshold(S, thr, cap=None):
    """Connected components of S > thr (= single linkage)."""
    A = csr_matrix(np.triu(S > thr, 1))
    _, lab = connected_components(A, directed=False)
    return cap_groups(lab, cap) if cap else lab


def mutual_knn(S, k, thr, cap=None):
    """Edges i-j when each is in the other's top-k and S_ij > thr; connected components."""
    n = len(S)
    T = S.copy()
    np.fill_diagonal(T, -np.inf)
    nn = np.argpartition(-T, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    M = np.zeros((n, n), dtype=bool)
    M[np.repeat(np.arange(n), k), nn.ravel()] = True
    E = M & M.T & (S > thr)
    _, lab = connected_components(csr_matrix(np.triu(E, 1)), directed=False)
    return cap_groups(lab, cap) if cap else lab


def leiden(S, k=10, res=1.0, thr=0.0, seed=0):
    """Leiden (CPM-free modularity) communities on the kNN graph (edge weight = similarity above thr)."""
    import igraph as ig
    import leidenalg
    n = len(S)
    T = S.copy()
    np.fill_diagonal(T, -np.inf)
    nn = np.argpartition(-T, kth=k - 1, axis=1)[:, :k]
    r = np.repeat(np.arange(n), k)
    c = nn.ravel()
    w = S[r, c]
    keep = w > thr
    r, c, w = r[keep], c[keep], w[keep]
    a, bb = np.minimum(r, c), np.maximum(r, c)
    key = a * n + bb
    _, ui = np.unique(key, return_index=True)
    g = ig.Graph(n=n, edges=list(zip(a[ui], bb[ui])), edge_attrs={'weight': w[ui]})
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight',
                                    resolution_parameter=res, seed=seed)
    return np.array(part.membership)


def hdbscan(S, min_cluster_size=2, min_samples=1, eps=0.0):
    from sklearn.cluster import HDBSCAN
    D = to_dist(S).astype(np.float64)
    D = (D + D.T) / 2
    lab = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed',
                  cluster_selection_epsilon=eps).fit_predict(D)
    noise = lab < 0
    lab = lab.copy()
    lab[noise] = lab.max() + 1 + np.arange(noise.sum())
    return lab


def auto_threshold(S, method='gmm'):
    """Pool-calibrated similarity threshold from the distribution of each passage's nearest-neighbour similarity:
    passages with a same-prompt partner form a high mode, singletons a low mode. Returns the boundary."""
    T = S.copy()
    np.fill_diagonal(T, -np.inf)
    nn = T.max(1).astype(np.float64)
    if method == 'gmm':
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(2, random_state=0).fit(nn[:, None])
        lo, hi = np.argsort(gm.means_.ravel())
        grid = np.linspace(nn.min(), nn.max(), 2000)
        p = gm.predict_proba(grid[:, None])
        cross = grid[(p[:, hi] > p[:, lo]) & (grid > gm.means_[lo, 0])]
        return float(cross[0]) if len(cross) else float(np.median(nn))
    if method == 'otsu':
        h, e = np.histogram(nn, 256)
        c = (e[:-1] + e[1:]) / 2
        w0 = np.cumsum(h)
        w1 = w0[-1] - w0
        m0 = np.cumsum(h * c) / np.maximum(w0, 1)
        m1 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w1, 1)
        return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])
    raise ValueError(method)
