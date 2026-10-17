"""Class-conditional n-gram language models (PPM-style compression authorship, Peng et al. 2003 / Teahan 2003).

For every class c, an interpolated Witten-Bell n-gram model (= PPM-C without exclusion) over a symbol sequence
(characters, bytes or LLM-tokenizer ids) is estimated from the training-fold passages of c. A held-out passage x
gets score_c(x) = log P_c(x) (total, or per symbol). argmax_c = minimum cross-entropy = best 'compressor'.
Fully vectorised: n-grams are rolling 64-bit hashes; lookups via np.searchsorted.

  python charlm.py <split> <variant> [<variant> ...]
variants: char<k>[_lc][_mean]  (k = max order), tokq<k>, tokg<k>, pmix<k>, ...
"""
import json
import sys
import time

import numpy as np

from common import CACHE, K, N, TEXTS, evaluate, get_folds, load_bases, load_oof, log_result, save_oof, y

P1 = np.uint64(1000003)
MASK_PAD = np.uint64(0)


def seq_chars(lower=False):
    vocab = {}
    out = []
    for t in TEXTS:
        t = t.lower() if lower else t
        out.append(np.array([vocab.setdefault(ch, len(vocab) + 1) for ch in t], dtype=np.uint64))
    return out, len(vocab) + 1


def seq_tokens(name):
    seqs = json.loads((CACHE / f'{name}.json').read_text(encoding='utf-8'))
    vocab = {}
    out = [np.array([vocab.setdefault(s, len(vocab) + 1) for s in q.split()], dtype=np.uint64) for q in seqs]
    return out, len(vocab) + 1


def ngram_hashes(seq, n):
    """hash of the n-gram ending at each position i (positions < n-1 padded with 0 = BOS symbol)."""
    L = len(seq)
    pad = np.concatenate([np.zeros(n - 1, dtype=np.uint64), seq])
    h = np.zeros(L, dtype=np.uint64)
    with np.errstate(over='ignore'):
        for j in range(n):
            h = h * P1 + pad[j:j + L] + np.uint64(1)
    return h


class WBModel:
    def __init__(self, seqs, order, V):
        self.order, self.V = order, V
        self.tabs = []
        for n in range(1, order + 1):
            ng = np.concatenate([ngram_hashes(s, n) for s in seqs])
            u, c = np.unique(ng, return_counts=True)
            if n > 1:
                ctx = np.concatenate([_ctx_hash(s, n) for s in seqs])   # context of the n-gram ending at i
                first = np.unique(ng, return_index=True)[1]
                cu, cc = np.unique(ctx, return_counts=True)              # c(ctx)
                tu, tc = np.unique(ctx[first], return_counts=True)       # T(ctx): distinct types after ctx
                self.tabs.append((u, c, cu, cc, tu, tc))
            else:
                self.tabs.append((u, c, len(ng), len(u)))

    @staticmethod
    def _look(keys, vals, q):
        idx = np.searchsorted(keys, q)
        idx = np.minimum(idx, len(keys) - 1)
        hit = keys[idx] == q
        return np.where(hit, vals[idx], 0)

    def logprob_h(self, H):
        """per-position log P(sym_i | history) under interpolated Witten-Bell; H[n] = (ngram hashes, ctx hashes)."""
        u, c, tot, types = self.tabs[0]
        cnt = self._look(u, c, H[1][0]).astype(np.float64)
        p = (cnt + types * (1.0 / self.V)) / (tot + types)
        for n in range(2, self.order + 1):
            u, c, cu, cc, tu, tc = self.tabs[n - 1]
            hng, hctx = H[n]
            cng = self._look(u, c, hng).astype(np.float64)
            cctx = self._look(cu, cc, hctx).astype(np.float64)
            tctx = self._look(tu, tc, hctx).astype(np.float64)
            p = np.where(cctx > 0, (cng + tctx * p) / np.maximum(cctx + tctx, 1e-12), p)
        return np.log(p)


def hashes_for(seqs, order):
    H = {}
    for n in range(1, order + 1):
        H[n] = (np.concatenate([ngram_hashes(s, n) for s in seqs]),
                np.concatenate([_ctx_hash(s, n) for s in seqs]) if n > 1 else None)
    return H


def _ctx_hash(seq, n):
    """hash of the (n-1)-gram ending at position i-1 (the context for position i), BOS-padded."""
    s2 = np.concatenate([np.zeros(1, dtype=np.uint64), seq[:-1]]) if len(seq) else seq
    # shift with a BOS symbol 0 so that the context of position 0 is all-BOS
    h = ngram_hashes(s2, n - 1)
    return h


def run(variant, split, folds):
    lower = '_lc' in variant
    mean = '_mean' in variant
    base = variant.split('_')[0]
    kind = ''.join(ch for ch in base if not ch.isdigit())
    order = int(''.join(ch for ch in base if ch.isdigit()))
    if kind == 'char':
        seqs, V = seq_chars(lower)
    else:
        name = {'tokq': 'tok_qwen3', 'tokg': 'tok_glm4', 'toko': 'tok_o200k', 'tokge': 'tok_gemma3',
                'pmix': 'pos_mixed', 'ptag': 'pos_tag', 'pospx': 'posproxy'}[kind]
        if name == 'posproxy':
            from views import transformed
            vocab = {}
            seqs = [np.array([vocab.setdefault(s, len(vocab) + 1) for s in q.split()], dtype=np.uint64)
                    for q in transformed('posproxy')]
            V = len(vocab) + 1
        else:
            seqs, V = seq_tokens(name)
    S = np.zeros((N, K))
    for a, b in folds:
        H = hashes_for([seqs[i] for i in b], order)
        lens = np.array([len(seqs[i]) for i in b], dtype=np.float64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        for c in range(K):
            m = WBModel([seqs[i] for i in a if y[i] == c], order, V)
            lp = m.logprob_h(H)
            tot = np.add.reduceat(lp, starts)
            S[b, c] = tot / lens if mean else tot
    return S


if __name__ == '__main__':
    split = int(sys.argv[1])
    bases = load_bases()
    folds = get_folds(split)
    for v in sys.argv[2:]:
        t0 = time.time()
        name = f'lm_{v}'
        S = load_oof(name, split)
        if S is None:
            S = run(v, split, folds)
            save_oof(name, split, S)
        r = evaluate(S, f'{name} [s{split}] ({time.time() - t0:.0f}s)', bases, folds)
        r['cand'] = name
        log_result(r, split)
