"""H1 step 4: is the sibling difference learnable? Binary A-vs-B classifiers on a paired, source-matched design.

For each contrast: pool prompts where both classes appear; per prompt pick one source shared by both
classes (random; fallback = any source), then one passage per class -> balanced, prompt-paired sample.
Subsample to <= N_PROMPTS prompts (CPU budget). char_wb 3-5 TF-IDF + logistic regression,
5-fold GroupKFold by prompt. Reports out-of-fold AUC and paired accuracy (share of prompts where the
B passage gets the higher P(B) than the A passage; 0.5 = indistinguishable).
Controls: cross-family anchor, within-family non-siblings, and same-model low-vs-default effort.
Writes h1_pairclf.csv.
"""
import time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
import sys
# usage: h1_pairclf.py [N_PROMPTS] [substring filter on contrast name, '|'-free, comma-separated] [out suffix]
N_PROMPTS = int(sys.argv[1]) if len(sys.argv) > 1 else 800
FILTER = sys.argv[2].split(",") if len(sys.argv) > 2 else None
SUFFIX = sys.argv[3] if len(sys.argv) > 3 else ""
rng = np.random.default_rng(0)
t0 = time.time()
pool = pd.read_parquet(ROOT / "research/ext_data/twostage/pool_v4.parquet")

def m(name): return pool.model == name
def ms(name, setting): return (pool.model == name) & (pool.setting == setting)
CONTRASTS = [
    ("sibling", "gemini-3.6-flash | gemini-3.7-flash", m("gemini-3.6-flash"), m("gemini-3.7-flash")),
    ("sibling", "gemini-3.6-flash | gemini-3.8-flash", m("gemini-3.6-flash"), m("gemini-3.8-flash")),
    ("sibling", "gemini-3.7-flash | gemini-3.8-flash", m("gemini-3.7-flash"), m("gemini-3.8-flash")),
    ("sibling", "glm-5.3 | glm-5.3-flash", m("glm-5.3"), m("glm-5.3-flash")),
    ("sibling", "gpt-5.6-luna | gpt-5.6-sol", m("gpt-5.6-luna"), m("gpt-5.6-sol")),
    ("sibling", "gpt-5.6-luna | gpt-5.6-terra", m("gpt-5.6-luna"), m("gpt-5.6-terra")),
    ("sibling", "gpt-5.6-sol | gpt-5.6-terra", m("gpt-5.6-sol"), m("gpt-5.6-terra")),
    ("sibling", "claude-opus-4-8 | claude-sonnet-5", m("claude-opus-4-8"), m("claude-sonnet-5")),
    ("control", "gpt-6-astra | claude-opus-5", m("gpt-6-astra"), m("claude-opus-5")),
    ("control", "gpt-5.5 | gpt-5.6-luna", m("gpt-5.5"), m("gpt-5.6-luna")),
    ("control", "claude-opus-4-6 | claude-opus-4-8", m("claude-opus-4-6"), m("claude-opus-4-8")),
    ("effort", "gpt-5.6-luna low | default", ms("gpt-5.6-luna", "low"), ms("gpt-5.6-luna", "default")),
    ("effort", "gemini-3.6-flash low | default", ms("gemini-3.6-flash", "low"), ms("gemini-3.6-flash", "default")),
    ("effort", "glm-5.3-flash low | default", ms("glm-5.3-flash", "low"), ms("glm-5.3-flash", "default")),
]

def paired_sample(ma, mb):
    A, B = pool[ma], pool[mb]
    common = np.intersect1d(A.prompt_key.unique(), B.prompt_key.unique())
    if len(common) > N_PROMPTS:
        common = rng.choice(common, N_PROMPTS, replace=False)
    ga = A[A.prompt_key.isin(common)].groupby("prompt_key")
    gb = B[B.prompt_key.isin(common)].groupby("prompt_key")
    rows, same_src = [], 0
    for pk in common:
        a, b = ga.get_group(pk), gb.get_group(pk)
        shared = np.intersect1d(a.source.unique(), b.source.unique())
        if len(shared):
            s = rng.choice(shared); a, b = a[a.source == s], b[b.source == s]; same_src += 1
        ra, rb = a.iloc[rng.integers(len(a))], b.iloc[rng.integers(len(b))]
        rows += [(pk, ra.text, 0), (pk, rb.text, 1)]
    return pd.DataFrame(rows, columns=["prompt_key", "text", "y"]), same_src / max(len(common), 1)

out = []
for kind, name, ma, mb in CONTRASTS:
    if FILTER and not any(f == name for f in FILTER):
        continue
    d, fss = paired_sample(ma, mb)
    if len(d) < 100:
        out.append(dict(kind=kind, contrast=name, n_prompts=len(d) // 2)); continue
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2).fit_transform(d.text)
    p = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(X, d.y, d.prompt_key):
        clf = LogisticRegression(C=4.0, max_iter=3000).fit(X[tr], d.y.iloc[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    d["p"] = p
    w = d.pivot(index="prompt_key", columns="y", values="p")
    paired = float((w[1] > w[0]).mean() + 0.5 * (w[1] == w[0]).mean())
    acc = float(((p > 0.5) == d.y).mean())
    out.append(dict(kind=kind, contrast=name, n_prompts=len(w), frac_same_source=round(fss, 3),
                    oof_auc=roc_auc_score(d.y, p), oof_acc=acc, paired_acc=paired))
    print(out[-1], f"{time.time()-t0:.0f}s", flush=True)
res = pd.DataFrame(out)
res.to_csv(OUT / f"h1_pairclf{SUFFIX}.csv", index=False, float_format="%.4f")
print(res.to_string(index=False))
