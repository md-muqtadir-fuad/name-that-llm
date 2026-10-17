"""Similarity-source specs -> pool similarity matrices, and algorithm specs -> labels.
Source spec grammar:
  base:   tfidf | tfidf12 | tfidf_stem | bm25 | jacc | q8L{4,7,12,16,22} | q8L7+22 | glm24 | <file>:<key>[:c]
          (<file>:<key> = cached embedding, e.g. q3e06:dtopic; ':c' = mean-centred over the pool before cosine)
  avg(a|b|w)        w * S_a + (1 - w) * S_b          (mixglm = avg(tfidf|glm24|0.5), the v1 notebook's signal)
  avg3(a|b|c)       equal-weight mean of three
  zavg(a|b|w)       same on globally standardised similarities
  rank(a|b)         mean of symmetrised within-row percentile ranks
  csls(a|k)         cross-domain similarity local scaling
  cl(a|lam)         cannot-link: S_a - lam * P(same predicted model) from the plain Qwen3-8B head
Algorithm spec: average | complete | single_cap6 | single_cap18 | mknn1 | mknn2 | leiden<res> | hdbscan |
  average_auto_gmm | average_auto_otsu  (thr ignored for the last three / leiden uses thr as edge floor)."""
import numpy as np

import methods as M
from common import FEATS, U, has_dist_q8, load_emb, q8_all

TEXTS = U['text'].values
_EMB = {}


def emb(name):
    if name not in _EMB:
        _EMB[name] = load_emb(name)
    return _EMB[name]


def _split(spec):
    inner = spec[spec.index('(') + 1:-1]
    parts, depth, cur = [], 0, ''
    for ch in inner:
        if ch == '|' and depth == 0:
            parts.append(cur)
            cur = ''
            continue
        depth += ch == '('
        depth -= ch == ')'
        cur += ch
    parts.append(cur)
    return spec[:spec.index('(')], parts


class Pool:
    def __init__(self, rows, psame=None):
        self.rows = rows
        self.cache = {}
        self.psame = psame   # callable returning P(same predicted model) matrix (for cl())

    def S(self, spec):
        if spec in self.cache:
            return self.cache[spec]
        S = self._make(spec)
        if len(self.cache) > 12:   # keep memory bounded (n^2 float32 each)
            self.cache.pop(next(iter(self.cache)))
        self.cache[spec] = S
        return S

    def _make(self, spec):
        rows = self.rows
        tx = list(TEXTS[rows])
        if '(' in spec:
            op, a = _split(spec)
            if op == 'avg':
                w = float(a[2])
                return w * self.S(a[0]) + (1 - w) * self.S(a[1])
            if op == 'zavg':
                w = float(a[2])
                return w * M.zsim(self.S(a[0])) + (1 - w) * M.zsim(self.S(a[1]))
            if op == 'avg3':
                return (self.S(a[0]) + self.S(a[1]) + self.S(a[2])) / 3
            if op == 'rank':
                return np.mean([M.rank_sim(self.S(x)) for x in a], 0).astype(np.float32)
            if op == 'csls':
                return M.csls(self.S(a[0]), int(a[1]))
            if op == 'cl':
                return self.S(a[0]) - float(a[1]) * self.psame()
            raise ValueError(spec)
        if spec == 'tfidf':
            return M.tfidf_sim(tx)
        if spec == 'tfidf12':
            return M.tfidf_sim(tx, ngram_range=(1, 2))
        if spec == 'tfidf_stem':
            return M.tfidf_stem_sim(tx)
        if spec == 'bm25':
            return M.bm25_sim(tx)
        if spec == 'jacc':
            return M.jaccard_sim(tx)
        if spec.startswith('q8L'):
            E = q8_all()
            lays = [4, 7, 12, 16, 22]
            Ls = [int(x) for x in spec[3:].split('+')]
            X = np.concatenate([E[rows, lays.index(L)].astype(np.float32) for L in Ls], 1)
            return M.cos_rows(X, center=True)
        if spec == 'glm24':
            return M.cos_rows(emb('glm24:E')[rows], center=True)
        parts = spec.split(':')
        X = emb(parts[0] + ':' + parts[1])[rows]
        return M.cos_rows(X, center=len(parts) > 2 and parts[2] == 'c')


def available_base():
    out = ['tfidf', 'tfidf12', 'tfidf_stem', 'bm25', 'jacc']
    if has_dist_q8():
        out += [f'q8L{L}' for L in (4, 7, 12, 16, 22)] + ['q8L7+22']
    if (FEATS / 'glm24.npz').exists():
        out += ['glm24']
    for f in sorted(FEATS.glob('*.npz')):
        if f.stem in ('q8_dist', 'glm24'):
            continue
        d = np.load(f)
        for k in d.files:
            if k != 'uid':
                out += [f'{f.stem}:{k}', f'{f.stem}:{k}:c']
    return out


def nn_grid(S, qs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    T = S.copy()
    np.fill_diagonal(T, -np.inf)
    return np.quantile(T.max(1), qs)


def run_algo(S, algo, thr, Zcache=None):
    if algo in ('average', 'complete'):
        key = algo
        if Zcache is not None and key in Zcache:
            Z = Zcache[key]
        else:
            Z = M.link(S, algo)
            if Zcache is not None:
                Zcache[key] = Z
        return M.cut(Z, thr)
    if algo == 'single_cap6':
        return M.cc_threshold(S, thr, cap=6)
    if algo == 'single_cap18':
        return M.cc_threshold(S, thr, cap=18)
    if algo.startswith('mknn'):
        return M.mutual_knn(S, int(algo[4:]), thr, cap=18)
    if algo.startswith('leiden'):
        return M.leiden(S, k=10, res=float(algo[6:]), thr=thr)
    if algo == 'hdbscan':
        return M.hdbscan(S)
    if algo.startswith('average_auto_'):
        t = M.auto_threshold(S, algo.split('_')[-1])
        return run_algo(S, 'average', t, Zcache)
    raise ValueError(algo)
