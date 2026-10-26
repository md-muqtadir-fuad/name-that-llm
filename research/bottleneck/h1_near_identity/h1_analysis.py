"""H1 step 2: sibling vs same-model vs cross-family similarity, per sibling pair.

Reads pairs.parquet (from h1_pairs.py). Writes:
  h1_pair_summary.csv   one row per (pair, metric): medians, AUC, closeness R with bootstrap CI
  h1_samemodel_split.csv cos_char medians of same-model pairs split by same/diff setting and source
  h1_train_view.csv     sibling vs cross-family cos_char on the 150 train prompts (no same-model there)

Definitions, for a pair (A,B), restricted to pool prompts where both A and B appear:
  S = A-vs-B pairs on the prompt; M = A-vs-A and B-vs-B pairs (repeated samples) on the prompt;
  C = {A or B} vs a model from another family on the prompt.
  Matched prompts = prompts with S, M and C all non-empty. Per prompt we average each set
  (s, m, c), then
  R = (mean s - mean c) / (mean m - mean c): 0 = siblings as different as cross-family,
      1 = siblings as similar as two samples of the same model.
  AUC = P(random M pair more similar than random S pair), pair level; 0.5 = indistinguishable.
  dz  = mean(m - s) / sd(m - s) over matched prompts (paired effect size, oriented so >0 means
        same-model more similar).
For claude-opus-4-8/claude-sonnet-5 neither model has repeats in the pool, so M is a PROXY:
same-model pairs of any other model on the same prompts (flagged in column m_source).
"""
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
P = pd.read_parquet(OUT / "pairs.parquet")
for c in ["model_i", "model_j", "source_i", "source_j", "setting_i", "setting_j", "prompt_key"]:
    P[c] = P[c].astype(str)
P["is_train"] = P.prompt_key.str.startswith("train:")

FAM = {"claude-fable-5": "claude", "claude-opus-4-6": "claude", "claude-opus-4-8": "claude",
       "claude-opus-5": "claude", "claude-sonnet-5": "claude",
       "gemini-3.1-pro-preview": "gemini", "gemini-3.6-flash": "gemini", "gemini-3.7-flash": "gemini",
       "gemini-3.8-flash": "gemini", "glm-5.3": "glm", "glm-5.3-flash": "glm",
       "gpt-5.5": "gpt", "gpt-5.6-luna": "gpt", "gpt-5.6-sol": "gpt", "gpt-5.6-terra": "gpt",
       "gpt-6-astra": "gpt", "kimi-k3": "kimi", "qwen3.8-max": "qwen"}
P["fam_i"] = P.model_i.map(FAM); P["fam_j"] = P.model_j.map(FAM)

SIB = [("gemini-3.6-flash", "gemini-3.7-flash"), ("gemini-3.6-flash", "gemini-3.8-flash"),
       ("gemini-3.7-flash", "gemini-3.8-flash"), ("glm-5.3", "glm-5.3-flash"),
       ("gpt-5.6-luna", "gpt-5.6-sol"), ("gpt-5.6-luna", "gpt-5.6-terra"),
       ("gpt-5.6-sol", "gpt-5.6-terra"), ("claude-opus-4-8", "claude-sonnet-5")]
CTRL = [("gpt-5.5", "gpt-5.6-luna"), ("gpt-6-astra", "gpt-5.6-sol"),
        ("gemini-3.1-pro-preview", "gemini-3.6-flash"), ("claude-opus-4-6", "claude-opus-4-8"),
        ("claude-opus-5", "claude-sonnet-5"), ("gpt-6-astra", "claude-opus-5")]
# metric -> +1 if larger = more similar, -1 if smaller = more similar
METRICS = {"cos_char": 1, "cos_word": 1, "len_ratio": 1, "sent_diff": -1, "ident_sent": 1}

pool = P[~P.is_train]
present = pool.groupby("prompt_key").apply(
    lambda d: set(d.model_i) | set(d.model_j), include_groups=False)


def auc(m, s):
    """P(m > s) + 0.5 P(m == s) via ranks."""
    m, s = np.asarray(m, float), np.asarray(s, float)
    if len(m) == 0 or len(s) == 0:
        return np.nan
    r = pd.Series(np.concatenate([m, s])).rank().to_numpy()
    return (r[:len(m)].sum() - len(m) * (len(m) + 1) / 2) / (len(m) * len(s))


rows, split_rows, train_rows = [], [], []
for kind, pairs in [("sibling", SIB), ("control", CTRL)]:
    for A, B in pairs:
        both = present[present.apply(lambda s: A in s and B in s)].index
        d = pool[pool.prompt_key.isin(both)]
        S = d[((d.model_i == A) & (d.model_j == B)) | ((d.model_i == B) & (d.model_j == A))]
        M = d[(d.model_i == d.model_j) & d.model_i.isin([A, B])]
        m_source = "own repeats"
        if len(M) == 0:
            M = d[d.model_i == d.model_j]
            m_source = "PROXY: other models' repeats, same prompts"
        C = d[((d.model_i.isin([A, B])) & (d.fam_j != FAM[A]) & (d.fam_j != FAM[B])) |
              ((d.model_j.isin([A, B])) & (d.fam_i != FAM[A]) & (d.fam_i != FAM[B]))]
        matched = sorted(set(S.prompt_key) & set(M.prompt_key) & set(C.prompt_key))
        Sm, Mm, Cm = (x[x.prompt_key.isin(matched)] for x in (S, M, C))
        for met, sign in METRICS.items():
            s = Sm.groupby("prompt_key")[met].mean().reindex(matched).to_numpy()
            m = Mm.groupby("prompt_key")[met].mean().reindex(matched).to_numpy()
            c = Cm.groupby("prompt_key")[met].mean().reindex(matched).to_numpy()
            R = (s.mean() - c.mean()) / (m.mean() - c.mean())
            bi = rng.integers(0, len(matched), (1000, len(matched)))
            Rb = (s[bi].mean(1) - c[bi].mean(1)) / (m[bi].mean(1) - c[bi].mean(1))
            diff = sign * (m - s)
            rows.append(dict(kind=kind, pair=f"{A} | {B}", metric=met, m_source=m_source,
                             n_prompts_both=len(both), n_prompts_matched=len(matched),
                             n_S=len(Sm), n_M=len(Mm), n_C=len(Cm),
                             med_S=Sm[met].median(), med_M=Mm[met].median(), med_C=Cm[met].median(),
                             auc_M_gt_S=auc(sign * Mm[met], sign * Sm[met]),
                             auc_S_gt_C=auc(sign * Sm[met], sign * Cm[met]),
                             R=R, R_lo=np.nanpercentile(Rb, 2.5), R_hi=np.nanpercentile(Rb, 97.5),
                             dz=diff.mean() / diff.std(ddof=1),
                             frac_prompts_S_ge_M=float((sign * (s - m) >= 0).mean()),
                             S_gt095=int((Sm.cos_char > 0.95).sum()) if met == "cos_char" else np.nan,
                             M_gt095=int((Mm.cos_char > 0.95).sum()) if met == "cos_char" else np.nan))
        # robustness: same-source only (S and M both restricted to same-source pairs)
        Ss, Ms = Sm[Sm.source_i == Sm.source_j], Mm[Mm.source_i == Mm.source_j]
        mt2 = sorted(set(Ss.prompt_key) & set(Ms.prompt_key))
        if mt2:
            s = Ss.groupby("prompt_key").cos_char.mean().reindex(mt2).to_numpy()
            m = Ms.groupby("prompt_key").cos_char.mean().reindex(mt2).to_numpy()
            c = Cm.groupby("prompt_key").cos_char.mean().reindex(mt2).to_numpy()
            ok = ~np.isnan(c)
            R2 = (s[ok].mean() - c[ok].mean()) / (m[ok].mean() - c[ok].mean())
        else:
            R2, ok = np.nan, np.array([])
        # condition-matched robustness: S and M both restricted to pairs that differ in source or
        # setting (xcond), or both restricted to same source AND same setting (samecond)
        def R_of(Sx, Mx):
            mt = sorted(set(Sx.prompt_key) & set(Mx.prompt_key))
            if not mt:
                return np.nan, 0, np.nan
            s = Sx.groupby("prompt_key").cos_char.mean().reindex(mt).to_numpy()
            m = Mx.groupby("prompt_key").cos_char.mean().reindex(mt).to_numpy()
            c = Cm.groupby("prompt_key").cos_char.mean().reindex(mt).to_numpy()
            return (s.mean() - c.mean()) / (m.mean() - c.mean()), len(mt), auc(Mx.cos_char, Sx.cos_char)
        xc = lambda X: X[(X.source_i != X.source_j) | (X.setting_i != X.setting_j)]
        sc = lambda X: X[(X.source_i == X.source_j) & (X.setting_i == X.setting_j)]
        R_x, n_x, auc_x = R_of(xc(Sm), xc(Mm))
        R_s, n_s, auc_s = R_of(sc(Sm), sc(Mm))
        # same-model split by setting/source
        split_rows.append(dict(kind=kind, pair=f"{A} | {B}", m_source=m_source,
                               R_xcond=R_x, auc_M_gt_S_xcond=auc_x, n_prompts_xcond=n_x,
                               R_samecond=R_s, auc_M_gt_S_samecond=auc_s, n_prompts_samecond=n_s,
                               S_xcond=xc(Sm).cos_char.median(), M_xcond=xc(Mm).cos_char.median(),
                               S_samecond=sc(Sm).cos_char.median(),
                               S_all=Sm.cos_char.median(),
                               S_same_source=Ss.cos_char.median(),
                               M_all=Mm.cos_char.median(),
                               M_same_src_same_set=Mm[(Mm.source_i == Mm.source_j) & (Mm.setting_i == Mm.setting_j)].cos_char.median(),
                               n_M_same_src_same_set=int(((Mm.source_i == Mm.source_j) & (Mm.setting_i == Mm.setting_j)).sum()),
                               M_diff_setting=Mm[Mm.setting_i != Mm.setting_j].cos_char.median(),
                               M_diff_source=Mm[Mm.source_i != Mm.source_j].cos_char.median(),
                               C_all=Cm.cos_char.median(),
                               R_same_source=R2, n_prompts_same_source=int(ok.sum())))
        # train view
        t = P[P.is_train]
        St = t[((t.model_i == A) & (t.model_j == B)) | ((t.model_i == B) & (t.model_j == A))]
        Ct = t[((t.model_i.isin([A, B])) & (t.fam_j != FAM[A]) & (t.fam_j != FAM[B])) |
               ((t.model_j.isin([A, B])) & (t.fam_i != FAM[A]) & (t.fam_i != FAM[B]))]
        train_rows.append(dict(kind=kind, pair=f"{A} | {B}", n_S=len(St),
                               train_med_S=St.cos_char.median(), train_med_C=Ct.cos_char.median(),
                               train_S_gt095=int((St.cos_char > 0.95).sum()),
                               train_ident_sent_S=St.ident_sent.mean(),
                               train_ident_sent_C=Ct.ident_sent.mean()))

res = pd.DataFrame(rows)
res.to_csv(OUT / "h1_pair_summary.csv", index=False, float_format="%.4f")
pd.DataFrame(split_rows).to_csv(OUT / "h1_samemodel_split.csv", index=False, float_format="%.4f")
pd.DataFrame(train_rows).to_csv(OUT / "h1_train_view.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(res[res.metric == "cos_char"].drop(columns=["metric"]).sort_values("R", ascending=False).to_string(index=False))
for met in ["cos_word", "len_ratio", "sent_diff", "ident_sent"]:
    print("\n", met)
    print(res[res.metric == met][["pair", "med_S", "med_M", "med_C", "auc_M_gt_S", "R", "R_lo", "R_hi", "dz"]].to_string(index=False))
print(pd.DataFrame(split_rows).to_string(index=False))
print(pd.DataFrame(train_rows).to_string(index=False))
