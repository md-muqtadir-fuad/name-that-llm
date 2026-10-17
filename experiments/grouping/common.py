"""Shared pieces of the test-grouping harness.

Pools: a simulated hidden test = train passages (all 150 prompts, k ~ U{lo..hi} per prompt) + distractor passages from
external corpora with their own prompts (extra-api: all 1000 prompts, k ~ U{lo..hi} each, capped at 5; oleg chat corpus
without refusals / speechmap prompts: 646 prompts, k capped at what exists, mostly 1). At 1-3/prompt: ~3000 passages,
~1800 prompts.
Grouping methods only see the pool's unlabelled text / embeddings. Pair metrics are computed on train rows (pairs with
at least one train row; train-distractor links are always false) and on all pool pairs.

Downstream (same twin pipeline as experiments/papers/paper_tests.py): frozen Qwen3-8B-Base L7+22 mean pools,
plain head = StandardScaler + LR(C=0.0025) fitted on the other 4 folds' prompts, applied with test-set standardisation;
centred twin = same LR on train rows centred by random same-prompt chunks of 2-3; on pool rows in a recovered group of
size 2..cap the twin's log-probs on group-centred features are ADDED (weight 1) to the plain log-probs.
Each train row is scored by the fold model that did not see its prompt (prompt-disjoint), so the pool can mix folds.
Test-set standardisation statistics come from the pool's TRAIN rows only (they mimic the 18-model test mix; the
distractors are 5-Claude-model / chat-corpus passages whose style would bias the statistics)."""
import pathlib
import pickle
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
GDIR = ROOT / 'experiments/grouping'
FEATS = GDIR / 'feats'
OUT = GDIR / 'out'
OUT.mkdir(exist_ok=True)

U = pd.read_parquet(FEATS / 'universe.parquet')
TR = pd.read_csv(ROOT / 'data/train.csv')
NTR = len(TR)
assert (U['uid'].values[:NTR] == ('train:' + TR['id'].astype(str)).values).all()
LABELS = sorted(TR['model'].unique())
K = len(LABELS)
Y = TR['model'].map({l: i for i, l in enumerate(LABELS)}).values
GTR = TR['prompt_id'].values
SRC = U['src'].values
PKEY = U['pkey'].values
PKEY_CODE = pd.factorize(PKEY)[0]
EXTRA_ROWS = np.where(SRC == 'extra')[0]
_REFUSAL = (r"^(?:I cannot|I can.t|I.m sorry|I won.t|I am unable|I.m unable|I.m not able|I am not able|I must decline|"
            r"Sorry|I apologize|I do not|I don.t feel|I.m not going|I will not)")
_SPEECHMAP = r"As a supporter of|Draft a passionate speech|persuasive essay|funny satirical"
# oleg chat-corpus distractors without refusals / speechmap provocation prompts (551 near-identical refusals would
# form a degenerate cluster that cannot occur in the competition test)
OLEG_ROWS = np.where((SRC == 'oleg') & ~U['text'].str.contains(_REFUSAL, regex=True).values
                     & ~U['pkey'].str.contains(_SPEECHMAP, regex=True).values)[0]


def splits():
    """A = GroupKFold(5) default (comparable with prior experiments); B = GroupKFold(5) on a shuffled prompt order."""
    A = list(GroupKFold(5).split(TR, Y, GTR))
    perm = np.random.RandomState(7).permutation(np.unique(GTR))
    code = pd.Series(np.arange(len(perm)), index=perm)[GTR].values
    B = list(GroupKFold(5).split(TR, Y, code))
    return {'A': A, 'B': B}


# ------------------------------------------------------------------------------------------ features
_Q8 = None


def q8_all():
    """Qwen3-8B-Base mean pools (layers 4,7,12,16,22) for the whole universe (n x 5 x 4096 fp16).
    Train from the earlier cache; distractors from the grouping feature-cache kernel (NaN if not available)."""
    global _Q8
    if _Q8 is not None:
        return _Q8
    z = np.load(ROOT / 'experiments/papers/qwen3_8b_train_feats.npz')
    assert (z['ids'] == TR['id'].values.astype(str)).all()
    E = np.full((len(U), 5, 4096), np.nan, dtype=np.float16)
    E[:NTR] = z['E']
    p = FEATS / 'q8_dist.npz'
    if p.exists():
        d = np.load(p)
        pos = pd.Series(np.arange(len(U)), index=U['uid'].values)
        uid = d['uid'].astype(str)
        dist = ~pd.Series(uid).str.startswith('train:').values
        E[pos[uid[dist]].values] = d['E'][dist]
    _Q8 = E
    return E


def has_dist_q8():
    return (FEATS / 'q8_dist.npz').exists()


def x722(rows):
    E = q8_all()
    return np.concatenate([E[rows, 1], E[rows, 4]], 1).astype(np.float32)   # layers 7, 22


def load_emb(name):
    """Embedding matrix aligned to U for a cached embedding '<file>:<key>' (fp32, NaN rows where missing)."""
    f, k = name.split(':')
    d = np.load(FEATS / f'{f}.npz')
    uid = d['uid'].astype(str)
    X = d[k].astype(np.float32)
    pos = pd.Series(np.arange(len(uid)), index=uid)
    return X[pos[U['uid'].values].values]


# ------------------------------------------------------------------------------------------ fold models
def fit_lr(X, y, C=0.0025):
    return LogisticRegression(C=C, max_iter=3000).fit(X, y)


def fold_models(split='A'):
    """Per fold: (plain scaler, plain LR, twin scaler, twin LR) on L7+22 of the fold's training prompts. Cached."""
    p = OUT / f'fold_models_{split}.pkl'
    if p.exists():
        return pickle.load(open(p, 'rb'))
    folds = splits()[split]
    X = x722(np.arange(NTR))
    rng = np.random.RandomState(0)
    res = []
    for fi, (a, b) in enumerate(folds):
        t0 = time.time()
        sc = StandardScaler().fit(X[a])
        clf = fit_lr(sc.transform(X[a]), Y[a])
        Xa = X[a]
        Xc = np.empty_like(Xa)
        for pid in np.unique(GTR[a]):
            idx = np.where(GTR[a] == pid)[0]
            rng.shuffle(idx)
            for c0 in range(0, len(idx), 3):
                ch = idx[c0:c0 + 3] if len(idx) - c0 > 3 else idx[c0:]
                Xc[ch] = Xa[ch] - Xa[ch].mean(0)
        tsc = StandardScaler().fit(Xc)
        tclf = fit_lr(tsc.transform(Xc), Y[a])
        res.append(dict(train=a, test=b, sc=sc, clf=clf, tsc=tsc, tclf=tclf))
        print(f'fold model {split}{fi} {time.time() - t0:.0f}s', flush=True)
    pickle.dump(res, open(p, 'wb'))
    return res


# ------------------------------------------------------------------------------------------ pools
def sample_pool(seed, lo=1, hi=3, n_oleg_prompts=10**6, distractors=True):
    """Row indices into U for one simulated test pool."""
    rng = np.random.RandomState(seed)
    rows = []
    for src_rows in ([np.arange(NTR)] + ([EXTRA_ROWS] if distractors else [])):
        r = src_rows
        codes = PKEY_CODE[r]
        for c in np.unique(codes):
            idx = r[codes == c]
            k = min(rng.randint(lo, hi + 1), len(idx))
            rows.extend(rng.choice(idx, k, replace=False))
    if distractors and n_oleg_prompts:
        codes = PKEY_CODE[OLEG_ROWS]
        pick = rng.choice(np.unique(codes), min(n_oleg_prompts, len(np.unique(codes))), replace=False)
        for c in pick:
            idx = OLEG_ROWS[codes == c]
            k = min(rng.randint(lo, hi + 1), len(idx))
            rows.extend(rng.choice(idx, k, replace=False))
    return np.sort(np.array(rows))


# ------------------------------------------------------------------------------------------ metrics
def pair_metrics(rows, lab):
    """Pair precision/recall/F1 on train rows (pairs with >= 1 train row) and on all pool pairs; grouped-share stats."""
    lab = np.asarray(lab)
    tr_mask = SRC[rows] == 'train'
    pk = PKEY_CODE[rows]
    out = {}
    # all pairs via contingency counting
    cl = pd.factorize(lab)[0]
    df = pd.DataFrame({'c': cl, 'p': pk, 't': tr_mask})

    def npairs(s):
        s = np.asarray(s, dtype=np.int64)
        return int((s * (s - 1) // 2).sum())
    tp_all = npairs(df.groupby(['c', 'p']).size())
    pred_all = npairs(df.groupby('c').size())
    true_all = npairs(df.groupby('p').size())
    # pairs with >= 1 train row: total - pairs among non-train rows
    nt = df[~df.t]
    tp_nt = npairs(nt.groupby(['c', 'p']).size())
    pred_nt = npairs(nt.groupby('c').size())
    true_nt = npairs(nt.groupby('p').size())
    tp_t, pred_t, true_t = tp_all - tp_nt, pred_all - pred_nt, true_all - true_nt
    # pairs with >= 1 extra-api row and no train row (extra prompts mimic the competition prompt mix but were
    # deduplicated against TRAIN topics, so train-row precision is optimistic; extra-row metrics show collisions
    # among test-like prompts)
    xs = SRC[rows] == 'extra'
    ox = df[~df.t & ~xs]
    tp_ox = npairs(ox.groupby(['c', 'p']).size())
    pred_ox = npairs(ox.groupby('c').size())
    true_ox = npairs(ox.groupby('p').size())
    tp_x, pred_x, true_x = tp_nt - tp_ox, pred_nt - pred_ox, true_nt - true_ox
    for nm, tp, pr, tru in (('t', tp_t, pred_t, true_t), ('x', tp_x, pred_x, true_x), ('all', tp_all, pred_all, true_all)):
        P = tp / pr if pr else 1.0
        R = tp / tru if tru else 1.0
        out[f'P_{nm}'], out[f'R_{nm}'], out[f'F1_{nm}'] = P, R, (2 * P * R / (P + R) if P + R else 0.0)
    sz = pd.Series(cl).map(pd.Series(cl).value_counts()).values
    out['grp_t'] = float((sz[tr_mask] >= 2).mean())
    # train rows whose group (size >= 2) is entirely one prompt
    pure = df.groupby('c')['p'].nunique()
    out['pure_t'] = float(((sz[tr_mask] >= 2) & (pure[cl[tr_mask]].values == 1)).mean())
    out['n_pool'] = len(rows)
    out['n_grp_pool'] = float((sz >= 2).mean())
    return out


class Downstream:
    """Plain + centred-twin evaluation on the train rows of a pool, for any grouping of the pool.

    dist: how distractor members (no cached Qwen3-8B features) enter a recovered group's centring mean.
      'ignore'    - left out of the mean (the original harness). A train row merged with distractors is then centred
                    only by its train co-members, so wrong merges with distractors cost NOTHING downstream: optimistic.
      'surrogate' - each distractor member that shares a group with >= 1 train row gets the features of a held-back
                    train passage (not in the pool) of a surrogate prompt; all passages of one distractor prompt map to
                    the same surrogate prompt, chosen (per group) to differ from every train prompt in that group.
                    So a wrong merge costs what merging with another prompt's passages costs (labels are used only
                    here, on the evaluation side - never by the grouping methods)."""

    def __init__(self, rows, split='A', models=None, dist='surrogate', sur_seed=0):
        self.rows = rows
        self.dist = dist
        self.models = models if models is not None else fold_models(split)
        folds = splits()[split]
        self.fold_of = np.full(NTR, -1)
        for fi, (a, b) in enumerate(folds):
            self.fold_of[b] = fi
        self.tr_pos = np.where(SRC[rows] == 'train')[0]          # positions in pool of train rows
        self.tr_rows = rows[self.tr_pos]
        self.y = Y[self.tr_rows]
        X = x722(rows)                                            # pool features (NaN for distractors w/o features)
        self.has = np.isfinite(X).all(1)
        self.X = np.where(self.has[:, None], X, 0)
        # surrogate plan for featureless distractors
        rng = np.random.RandomState(10007 + sur_seed)
        in_pool = np.zeros(NTR, bool)
        in_pool[self.tr_rows] = True
        prompts = np.unique(GTR)
        self._avail = {p: rng.permutation(np.where((GTR == p) & ~in_pool)[0]) for p in prompts}
        self._dpos = np.where(~self.has)[0]
        dkey = PKEY_CODE[rows[self._dpos]]
        self._dkey = dkey
        self._drank = pd.Series(dkey).groupby(dkey).cumcount().values
        self._cands = {k: rng.permutation(prompts) for k in np.unique(dkey)}
        Xt = self.X[self.tr_pos]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6                    # test-std stats from train rows of the pool
        self.mu, self.sd = mu, sd
        self.Lp = np.zeros((len(self.tr_pos), K))
        self.fo = self.fold_of[self.tr_rows]
        for fi, m in enumerate(self.models):
            i = self.fo == fi
            self.Lp[i] = m['clf'].predict_log_proba((Xt[i] - mu) / sd)
        self.plain = bacc(self.y, self.Lp.argmax(1))

    def _surrogate(self, lab):
        """(positions, features) of the distractor members that share a group with a train row."""
        tr_lab = lab[self.tr_pos]
        tr_p = GTR[self.tr_rows]
        need = np.isin(lab[self._dpos], tr_lab)
        pos = self._dpos[need]
        if not len(pos):
            return pos, np.zeros((0, self.X.shape[1]), np.float32)
        gp = pd.Series(tr_p).groupby(tr_lab).agg(set).to_dict()     # group -> train prompts in it
        src = []
        for p_, k, r in zip(pos, self._dkey[need], self._drank[need]):
            bad = gp.get(lab[p_], set())
            sp = next(c for c in self._cands[k] if c not in bad)
            av = self._avail[sp]
            src.append(av[r % len(av)])
        return pos, x722(np.array(src))

    def evaluate(self, lab, cap=18, w=1.0, min_size=2, return_pred=False, row_w=None):
        """Balanced accuracy on the pool's train rows with the twin added on groups of size min_size..cap.
        Group means use every member that has features (distractor members included when cached)."""
        lab = pd.factorize(np.asarray(lab))[0]
        n = len(lab)
        M = csr_matrix((self.has.astype(np.float32), (lab, np.arange(n))), shape=(lab.max() + 1, n))
        cnt = np.asarray(M.sum(1)).ravel()
        sz = np.bincount(lab)
        sums = M @ self.X
        if self.dist == 'surrogate':
            pos, Xs = self._surrogate(lab)
            if len(pos):
                Ms = csr_matrix((np.ones(len(pos), np.float32), (lab[pos], np.arange(len(pos)))),
                                shape=(lab.max() + 1, len(pos)))
                sums = sums + Ms @ Xs
                cnt = cnt + np.asarray(Ms.sum(1)).ravel()
        means = sums / np.maximum(cnt, 1)[:, None]
        g = lab[self.tr_pos]
        use = (sz[g] >= min_size) & (sz[g] <= cap) & (cnt[g] >= 2)
        L = self.Lp.copy()
        if use.any():
            Xc = self.X[self.tr_pos[use]] - means[g[use]]
            fo = self.fo[use]
            Lt = np.zeros((use.sum(), K))
            for fi, m in enumerate(self.models):
                i = fo == fi
                if i.any():
                    Lt[i] = m['tclf'].predict_log_proba(m['tsc'].transform(Xc[i]))
            ww = w if row_w is None else w * np.asarray(row_w)[self.tr_pos[use]][:, None]
            L[use] += ww * Lt
        if return_pred:
            return L.argmax(1)
        return bacc(self.y, L.argmax(1)), float(use.mean())

    def psame(self):
        """P(two pool passages get the same predicted model) = p_i . p_j from the plain head (train rows: their own
        fold model; distractors: fold-0 model). Used as a soft cannot-link (one passage per model per prompt)."""
        P = np.zeros((len(self.rows), K))
        Z = (self.X - self.mu) / self.sd
        fo = np.full(len(self.rows), 0)
        fo[self.tr_pos] = self.fo
        for fi, m in enumerate(self.models):
            i = fo == fi
            P[i] = m['clf'].predict_proba(Z[i])
        P[~self.has] = 1.0 / K
        return (P @ P.T).astype(np.float32)

    def true_labels(self):
        return PKEY_CODE[self.rows]
