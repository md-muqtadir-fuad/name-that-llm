"""Step 1: stylometric EDA per model and per lab. Writes out/eda.md"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from common import OUT, LAB, CLASSES, load_train, stylo_features

df = load_train()
F = stylo_features(df.text)
F["model"] = df.model.values
F["lab"] = df.lab.values
feat_cols = [c for c in F.columns if c not in ("model", "lab")]
lines = []
P = lines.append

# ------------------------------------------------------------ feature ranking
P("## 1. Stylometric EDA\n")
P("### 1.1 Global discriminativeness (Kruskal-Wallis H across 18 models; eta^2 from one-way ANOVA)\n")
rank = []
for c in feat_cols:
    groups = [F.loc[F.model == m, c].values for m in CLASSES]
    try:
        H, p = stats.kruskal(*groups)
    except ValueError:
        H, p = 0.0, 1.0
    grand = F[c].mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    sst = ((F[c] - grand) ** 2).sum()
    rank.append((c, H, p, ssb / sst if sst > 0 else 0))
rank = pd.DataFrame(rank, columns=["feature", "H", "p", "eta2"]).sort_values("H", ascending=False)
P("| feature | H | eta^2 |\n|---|---|---|")
for _, r in rank.head(30).iterrows():
    P(f"| {r.feature} | {r.H:.0f} | {r.eta2:.3f} |")
P("")

# ------------------------------------------------------------- per-model table
key = ["n_words", "em_dash_pw", "en_dash_pw", "semicolon_pw", "colon_pw", "curly_apos_pw",
       "straight_apos_pw", "curly_dq_pw", "straight_dq_pw", "comma_pw", "words_per_sent",
       "mean_word_len", "type_token", "paren_pw", "question_pw", "exclam_pw", "you_pw",
       "not_x_but_y", "markdown_any", "has_newline"]
P("### 1.2 Per-model means (per-word rates x100 for *_pw)\n")
tab = F.groupby("model")[key].mean()
for c in key:
    if c.endswith("_pw"):
        tab[c] = tab[c] * 100
P("| model | " + " | ".join(key) + " |")
P("|" + "---|" * (len(key) + 1))
for m, r in tab.iterrows():
    P(f"| {m} | " + " | ".join(f"{v:.2f}" for v in r.values) + " |")
P("")

# usage rates (share of passages that contain the character at least once)
P("### 1.3 Share of passages containing a mark at least once (%)\n")
marks = {"em dash —": "—", "en dash –": "–", "semicolon": ";", "colon": ":",
         "curly apostrophe ’": "’", "straight apostrophe '": "'", "curly quotes “": "“",
         "straight quotes \"": '"', "parenthesis": "(", "question ?": "?", "exclamation !": "!",
         "spaced hyphen ' - '": " - ", "newline": "\n", "**": "**", "...": "..."}
use = pd.DataFrame({k: df.text.str.contains(re.escape(v)).groupby(df.model).mean() * 100 for k, v in marks.items()})
P("| model | " + " | ".join(use.columns) + " |")
P("|" + "---|" * (len(use.columns) + 1))
for m, r in use.iterrows():
    P(f"| {m} | " + " | ".join(f"{v:.0f}" for v in r.values) + " |")
P("")

# per lab
P("### 1.4 Same, per lab (%)\n")
usel = pd.DataFrame({k: df.text.str.contains(re.escape(v)).groupby(df.lab).mean() * 100 for k, v in marks.items()})
P("| lab | " + " | ".join(usel.columns) + " |")
P("|" + "---|" * (len(usel.columns) + 1))
for m, r in usel.iterrows():
    P(f"| {m} | " + " | ".join(f"{v:.0f}" for v in r.values) + " |")
P("")

# ------------------------------------------------------ most distinctive per model
P("### 1.5 Most distinctive stylometric features per model (standardized mean diff vs other 17 models)\n")
Z = (F[feat_cols] - F[feat_cols].mean()) / (F[feat_cols].std() + 1e-12)
P("| model | top + features (d) | top - features (d) |\n|---|---|---|")
for m in CLASSES:
    d = Z[F.model == m].mean() - Z[F.model != m].mean()
    d = d.drop(["n_chars"], errors="ignore")
    pos = d.sort_values(ascending=False).head(5)
    neg = d.sort_values().head(4)
    P(f"| {m} | " + ", ".join(f"{k} {v:+.2f}" for k, v in pos.items()) + " | " +
      ", ".join(f"{k} {v:+.2f}" for k, v in neg.items()) + " |")
P("")
P("### 1.6 Per lab\n")
P("| lab | top + features (d) | top - features (d) |\n|---|---|---|")
for lab in sorted(set(LAB.values())):
    d = Z[F.lab == lab].mean() - Z[F.lab != lab].mean()
    d = d.drop(["n_chars"], errors="ignore")
    pos = d.sort_values(ascending=False).head(5)
    neg = d.sort_values().head(4)
    P(f"| {lab} | " + ", ".join(f"{k} {v:+.2f}" for k, v in pos.items()) + " | " +
      ", ".join(f"{k} {v:+.2f}" for k, v in neg.items()) + " |")
P("")

# ------------------------------------------------------------ length vs model
P("### 1.7 Length: does it carry label signal?\n")
nw = F.n_words
P(f"Word count by model: eta^2 = {rank.set_index('feature').loc['n_words','eta2']:.3f}; per-model median range "
  f"{F.groupby('model').n_words.median().min():.0f}-{F.groupby('model').n_words.median().max():.0f} words.")
# within-prompt rank of length
df["_nw"] = nw.values
df["_rank"] = df.groupby("prompt_id")["_nw"].rank(pct=True)
r = df.groupby("model")["_rank"].mean().sort_values()
P("Mean within-prompt percentile rank of length (0.5 = no effect): " +
  ", ".join(f"{k} {v:.2f}" for k, v in r.items()))
# per-prompt length range
pr = df.groupby("prompt_id")["_nw"].agg(["min", "max"])
P(f"\nPer-prompt min/max words: median min {pr['min'].median():.0f}, median max {pr['max'].median():.0f}.\n")

# ------------------------------------------------------------ starting words
P("### 1.8 Opening words\n")
first2 = df.text.str.strip().str.split().str[:2].str.join(" ").str.lower()
first1 = df.text.str.strip().str.split().str[0].str.lower()
P("Top first word per model (share %):\n")
P("| model | most common first words |\n|---|---|")
for m in CLASSES:
    c = first1[df.model == m].value_counts(normalize=True).head(5)
    P(f"| {m} | " + ", ".join(f"{k} {v*100:.0f}" for k, v in c.items()) + " |")
P("")
# opening phrase shared across siblings
same_open = df.groupby("prompt_id").apply(lambda g: first2.loc[g.index].value_counts().iloc[0] / len(g))
P(f"Within a prompt, the most common 2-word opening is shared by on average {same_open.mean()*100:.0f}% of the 18 answers "
  f"(median {same_open.median()*100:.0f}%). => openings are prompt-driven, a clustering cue, weak label cue.\n")

# --------------------------------------------- word log-odds (Monroe et al.)
P("### 1.9 Most model-specific tokens (log-odds ratio with informative Dirichlet prior, z-score)\n")
tok = re.compile(r"[A-Za-z']+|[—–;:!?()’“”\"]")
docs = [tok.findall(t.lower()) for t in df.text]
glob = Counter()
for d in docs:
    glob.update(d)
vocab = [w for w, c in glob.items() if c >= 8]
a0 = sum(glob[w] for w in vocab)
alpha = {w: glob[w] / a0 * 500 for w in vocab}
A = sum(alpha.values())


def logodds(mask):
    ci, cj = Counter(), Counter()
    for d, mm in zip(docs, mask):
        (ci if mm else cj).update(d)
    ni, nj = sum(ci[w] for w in vocab), sum(cj[w] for w in vocab)
    out = {}
    for w in vocab:
        a = alpha[w]
        li = np.log((ci[w] + a) / (ni + A - ci[w] - a))
        lj = np.log((cj[w] + a) / (nj + A - cj[w] - a))
        var = 1 / (ci[w] + a) + 1 / (cj[w] + a)
        out[w] = (li - lj) / np.sqrt(var)
    return pd.Series(out).sort_values(ascending=False)


P("| model | most over-used tokens (z) | most under-used (z) |\n|---|---|---|")
for m in CLASSES:
    s = logodds((df.model == m).values)
    P(f"| {m} | " + ", ".join(f"`{k}` {v:.1f}" for k, v in s.head(10).items()) + " | " +
      ", ".join(f"`{k}` {v:.1f}" for k, v in s.tail(5)[::-1].items()) + " |")
P("")
P("| lab | most over-used tokens (z) | most under-used (z) |\n|---|---|---|")
for lab in sorted(set(LAB.values())):
    s = logodds((df.lab == lab).values)
    P(f"| {lab} | " + ", ".join(f"`{k}` {v:.1f}" for k, v in s.head(10).items()) + " | " +
      ", ".join(f"`{k}` {v:.1f}" for k, v in s.tail(5)[::-1].items()) + " |")
P("")

(OUT / "eda.md").write_text("\n".join(lines), encoding="utf-8")
F.to_parquet(OUT / "stylo_train.parquet")
print("\n".join(lines))
