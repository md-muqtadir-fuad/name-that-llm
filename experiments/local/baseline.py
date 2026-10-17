"""Step 2: TF-IDF baselines, 5-fold prompt-grouped CV. Saves OOF log-probs to out/oof_*.npy
and writes out/baseline.md
"""
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import OUT, CLASSES, K, LAB, load_train, folds, bacc, log_softmax, stylo_features

df = load_train()
y = df.y.values
fold = folds(df)
texts = df.text.values
STY = stylo_features(texts).drop(columns=["n_chars"]).values.astype(float)


def make_vecs(sublinear, analyzer="char_wb", lowercase=True):
    return [
        TfidfVectorizer(analyzer=analyzer, ngram_range=(1, 5), min_df=2, sublinear_tf=sublinear,
                        lowercase=lowercase, max_features=300000, dtype=np.float32),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=sublinear,
                        lowercase=lowercase, token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=200000,
                        dtype=np.float32),
    ]


def run_fold(k, cfg):
    tr, va = fold != k, fold == k
    vecs = make_vecs(cfg["sublinear"], cfg.get("analyzer", "char_wb"), cfg.get("lowercase", True))
    Xtr = [v.fit_transform(texts[tr]) for v in vecs]
    Xva = [v.transform(texts[va]) for v in vecs]
    if cfg.get("stylo"):
        sc = StandardScaler().fit(STY[tr])
        w = cfg["stylo"]
        Xtr.append(sparse.csr_matrix(np.clip(sc.transform(STY[tr]), -5, 5) * w))
        Xva.append(sparse.csr_matrix(np.clip(sc.transform(STY[va]), -5, 5) * w))
    Xtr, Xva = sparse.hstack(Xtr).tocsr(), sparse.hstack(Xva).tocsr()
    if cfg["clf"] == "lr":
        clf = LogisticRegression(C=cfg["C"], max_iter=3000)
        clf.fit(Xtr, y[tr])
        s = clf.predict_log_proba(Xva)
    else:
        clf = LinearSVC(C=cfg["C"])
        clf.fit(Xtr, y[tr])
        s = clf.decision_function(Xva)
    return k, np.where(va)[0], s


def evaluate(name, cfg):
    t = time.time()
    res = Parallel(n_jobs=4)(delayed(run_fold)(k, cfg) for k in range(5))
    S = np.zeros((len(df), K))
    for k, idx, s in res:
        S[idx] = s
    # temperature for SVM scores (1 scalar fitted on OOF NLL; monotone -> does not change argmax)
    if cfg["clf"] == "svm":
        f = lambda T: -log_softmax(S * T)[np.arange(len(y)), y].mean()
        T = minimize_scalar(f, bounds=(0.1, 50), method="bounded").x
        L = log_softmax(S * T)
    else:
        T = 1.0
        L = log_softmax(S)
    pred = L.argmax(1)
    ba = bacc(y, pred)
    per_fold = [bacc(y[fold == k], pred[fold == k]) for k in range(5)]
    labs = np.array([LAB[c] for c in CLASSES])
    lab_ba = bacc(labs[y], labs[pred])
    nll = -L[np.arange(len(y)), y].mean()
    np.save(OUT / f"oof_{name}.npy", L.astype(np.float32))
    print(f"{name}: bacc {ba:.4f} (folds {np.std(per_fold):.3f} sd) lab-bacc {lab_ba:.4f} nll {nll:.3f} T={T:.2f} [{time.time()-t:.0f}s]", flush=True)
    return dict(name=name, bacc=ba, fold_sd=np.std(per_fold), lab_bacc=lab_ba, nll=nll, secs=time.time() - t, pred=pred)


CFGS = {
    "lr_raw":        dict(sublinear=False, clf="lr", C=10.0),
    "lr_sub":        dict(sublinear=True, clf="lr", C=10.0),
    "lr_sub_C30":    dict(sublinear=True, clf="lr", C=30.0),
    "svm_sub":       dict(sublinear=True, clf="svm", C=0.5),
    "svm_sub_C0.2":  dict(sublinear=True, clf="svm", C=0.2),
    "svm_sub_sty":   dict(sublinear=True, clf="svm", C=0.5, stylo=0.15),
    "lr_sub_sty":    dict(sublinear=True, clf="lr", C=10.0, stylo=0.15),
    "svm_char_case": dict(sublinear=True, clf="svm", C=0.5, analyzer="char", lowercase=False),
    "svm_char_case_sty": dict(sublinear=True, clf="svm", C=0.5, analyzer="char", lowercase=False, stylo=0.15),
}
if len(sys.argv) > 1:
    CFGS = {k: v for k, v in CFGS.items() if k in sys.argv[1:]}

results = [evaluate(n, c) for n, c in CFGS.items()]

lines = ["## 2. TF-IDF baselines (5-fold GroupKFold by prompt_id, OOF)\n",
         "Features: char_wb(1-5) + word(1-2, punctuation tokens kept), min_df=2, vectorizers fit on the train fold only.\n",
         "| variant | balanced acc | fold sd | lab-level bacc | NLL | secs |", "|---|---|---|---|---|---|"]
for r in results:
    lines.append(f"| {r['name']} | {r['bacc']:.4f} | {r['fold_sd']:.3f} | {r['lab_bacc']:.4f} | {r['nll']:.3f} | {r['secs']:.0f} |")
best = max(results, key=lambda r: r["bacc"])
pred = best["pred"]
cm = confusion_matrix(y, pred, labels=range(K))
rec = cm.diagonal() / cm.sum(1)
lines.append(f"\nBest: **{best['name']}**. Per-class recall:\n")
lines.append("| model | recall | most-confused-with (share of its passages) |\n|---|---|---|")
for i, c in enumerate(CLASSES):
    row = cm[i].astype(float) / cm[i].sum()
    row[i] = 0
    top = np.argsort(-row)[:3]
    lines.append(f"| {c} | {rec[i]:.2f} | " + ", ".join(f"{CLASSES[j]} {row[j]*100:.0f}%" for j in top) + " |")
pairs = []
for i in range(K):
    for j in range(i + 1, K):
        pairs.append((cm[i, j] + cm[j, i], CLASSES[i], CLASSES[j]))
pairs.sort(reverse=True)
lines.append("\nTop confused pairs (symmetric counts out of ~300):\n")
lines.append(", ".join(f"{a} <-> {b}: {n}" for n, a, b in pairs[:15]))
labs = np.array([LAB[c] for c in CLASSES])
within = np.mean(labs[pred[pred != y]] == labs[y[pred != y]])
lines.append(f"\nShare of errors that stay within the true lab: {within*100:.0f}%.\n")
(OUT / f"baseline{'_' + '_'.join(sys.argv[1:]) if len(sys.argv) > 1 else ''}.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
