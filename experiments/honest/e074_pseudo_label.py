"""E074 (test-time adaptation 2): pseudo-label self-training of a TF-IDF member on the test texts.
Per seeded fold k (held-out rows = the "test"): the honest blend (0.664 weights, members we can score, test-like 2-3
answer groups, Sinkhorn 0.45 on the fold slice) labels the test rows; the top-q most confident rows (q in {0.3, 0.6})
are added with their pseudo-labels to the training rows of a TF-IDF logistic regression (char_wb 2-5 + word 1-2, the
public tfidf member's views, fit on train + test texts, unsupervised). Control: the same model on the training rows only.
Each member then joins the blend with a weight picked on the other 4 folds from {0, 0.1, 0.25, 0.5, 1} and is scored
on fold k; 3 draws of the 2-3 groups (pseudo-labels from draw 0's blend). Run from the repo root: python experiments/honest/e074_pseudo_label.py [--time]"""
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bas

sys.path.insert(0, 'paper')
from log import log_row  # noqa: E402

src = open('experiments/honest/e069_bias.py', encoding='utf-8').read()
exec(src.split("rng = np.random.default_rng(0)\nres = []")[0])   # y, fold, pid, train, blend(), sinkhorn()
from scipy.special import softmax  # noqa: E402

text = train.text.fillna('').astype(str).to_numpy()
X = sp.hstack([TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=60000, lowercase=False).fit_transform(text),
               TfidfVectorizer(sublinear_tf=True, token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), min_df=3, max_features=30000, lowercase=False).fit_transform(text)]).tocsr()


def fit_predict(tr, te, extra_rows=None, extra_y=None):
    Xa, ya = X[tr], y[tr]
    if extra_rows is not None and len(extra_rows):
        Xa = sp.vstack([Xa, X[extra_rows]]); ya = np.concatenate([ya, extra_y])
    return LogisticRegression(C=20, max_iter=2000).fit(Xa, ya).predict_log_proba(X[te])


if '--time' in sys.argv:
    t0 = time.time(); fit_predict(np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)); print(f'one fit: {time.time() - t0:.0f} s'); sys.exit()

ctrl = np.zeros((len(y), 18))
for k in range(5):
    ctrl[fold == k] = fit_predict(np.flatnonzero(fold != k), np.flatnonzero(fold == k))
print(f'control member standalone {bas(y, ctrl.argmax(1)):.4f}', flush=True)
Q, WG, NB = (0.3, 0.6), (0.0, 0.1, 0.25, 0.5, 1.0), 11
rng = np.random.default_rng(0)
res = []
DRAWS = []
for d in range(3):
    g = np.full(len(y), -1); nxt = 0
    for p_ in np.unique(pid):
        ii = rng.permutation(np.flatnonzero(pid == p_)); i = 0
        while i < len(ii):
            n = 2 if len(ii) - i in (2, 4) else int(rng.integers(2, 4)); g[ii[i:i + n]] = nxt; nxt += 1; i += n
    DRAWS.append(blend(g, False))
P = np.zeros((len(y), 18))
for k in range(5):
    P[fold == k] = softmax(sinkhorn(DRAWS[0][fold == k]), 1)
mem = {'control': ctrl}
for q in Q:
    M = np.zeros((len(y), 18))
    for k in range(5):
        te = np.flatnonzero(fold == k)
        conf = P[te].max(1)
        sel = te[conf >= np.quantile(conf, 1 - q)]
        M[te] = fit_predict(np.flatnonzero(fold != k), te, sel, P[sel].argmax(1))
        print(f'  q {q} fold {k}: pseudo rows {len(sel)}, pseudo-label accuracy {(P[sel].argmax(1) == y[sel]).mean():.3f}', flush=True)
    mem[f'pseudo q={q}'] = M
for d in range(3):
    lpn = DRAWS[d]
    tot = lpn * NB
    for name, M in mem.items():
        pred = np.empty(len(y), int); picked = []
        for k in range(5):
            tr, te = fold != k, fold == k
            # weight choice on the other folds (each fold slice gets its own Sinkhorn, as at test time)
            sc = {}
            for w in WG:
                pr = np.empty(len(y), int)
                for j in range(5):
                    if j == k: continue
                    sl = fold == j
                    pr[sl] = sinkhorn((tot[sl] + w * M[sl]) / (NB + (w > 0))).argmax(1)
                sc[w] = bas(y[tr], pr[tr])
            w = max(sc, key=sc.get); picked.append(w)
            pred[te] = sinkhorn((tot[te] + w * M[te]) / (NB + (w > 0))).argmax(1)
        res.append(dict(draw=d, member=name, standalone=bas(y, M.argmax(1)), blend=bas(y, pred), weights=picked))
        print(f'draw {d} {name:14s} standalone {res[-1]["standalone"]:.4f} blend {res[-1]["blend"]:.4f} weights {picked}', flush=True)
R = pd.DataFrame(res)
Pv = R.pivot_table(index='draw', columns='member', values='blend')
print(R.groupby('member')[['standalone', 'blend']].mean().round(4).to_string())
out = []
for q in Q:
    dlt = Pv[f'pseudo q={q}'] - Pv['control']
    out.append(f"q={q}: {dlt.mean():+.4f} (se {dlt.std() / np.sqrt(len(dlt)):.4f}, {(dlt > 0).sum()}/{len(dlt)})")
print('pseudo vs control in blend:', '; '.join(out))
R.to_csv('experiments/honest/out/e074_pseudo_label.csv', index=False)
st = R.groupby('member').standalone.mean()
log_row(id='E074', category='tta', metric='bacc_honest_blend_seeded',
        description='test-time pseudo-label self-training: TF-IDF LR retrained with the blend-labelled most confident test rows (q of the fold) vs the same model without them, each added to the honest blend with a CV-picked weight (0 allowed)',
        config='views char_wb 2-5 (60k) + word 1-2 (30k), LR C=20; q in {0.3, 0.6}; weights {0,0.1,0.25,0.5,1} picked on 4 folds; Sinkhorn 0.45 per fold slice; 3 draws of 2-3 groups',
        baseline=f"{Pv['control'].mean():.4f}", value='; '.join(f"q={q} {Pv[f'pseudo q={q}'].mean():.4f}" for q in Q), folds_pos='',
        files='experiments/honest/out/e074_pseudo_label.csv',
        notes='standalone: ' + ', '.join(f'{k} {v:.4f}' for k, v in st.items()) + ' | blend diff vs control: ' + '; '.join(out))
