"""H1 step 3: global exact / near-duplicate search across ALL passages (pool + train), any prompt.

Candidates: pairs sharing word 6-gram shingles (shingles with doc-freq 2..50), containment >= 0.3.
Then char_wb 3-5 TF-IDF cosine (same settings as h1_pairs.py) on candidates; near-dup = cos > 0.95.
Writes h1_neardup_pairs.csv (all candidate pairs with cos > 0.80) and h1_neardup_counts.csv.
"""
import re, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
t0 = time.time()
pool = pd.read_parquet(ROOT / "research/ext_data/twostage/pool_v4.parquet")
tr = pd.read_csv(ROOT / "data/train.csv").assign(source="train", setting="train")
tr["prompt_key"] = "train:" + tr.prompt_id.astype(str)
cols = ["id", "prompt_key", "text", "model", "source", "setting"]
df = pd.concat([pool[cols], tr[cols]], ignore_index=True)

WORD = re.compile(r"\w+")
recs_d, recs_h, nsh = [], [], np.zeros(len(df), int)
for k, t in enumerate(df.text):
    w = WORD.findall(t.lower())
    sh = {hash(tuple(w[i:i + 6])) for i in range(max(0, len(w) - 5))}
    nsh[k] = len(sh)
    recs_d.extend([k] * len(sh)); recs_h.extend(sh)
S = pd.DataFrame({"d": np.array(recs_d, np.int32), "h": np.array(recs_h, np.int64)})
dfreq = S.h.map(S.h.value_counts())
S = S[(dfreq >= 2) & (dfreq <= 50)]
print("shingle rows", len(S), f"{time.time()-t0:.0f}s", flush=True)
Mg = S.merge(S, on="h")
Mg = Mg[Mg.d_x < Mg.d_y]
C = Mg.groupby(["d_x", "d_y"]).size().rename("shared").reset_index()
C["containment"] = C.shared / np.minimum(nsh[C.d_x], nsh[C.d_y]).clip(min=1)
C = C[C.containment >= 0.3].reset_index(drop=True)
print("candidates", len(C), f"{time.time()-t0:.0f}s", flush=True)

ids = np.unique(np.concatenate([C.d_x, C.d_y]))
hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**21,
                       alternate_sign=False, norm=None)
# idf fit on a 15k random subsample of the corpus (CPU budget); near-dup threshold is insensitive to it
tf = TfidfTransformer(sublinear_tf=True).fit(hv.transform(df.text.sample(15000, random_state=0)))
X = tf.transform(hv.transform(df.text.iloc[ids]))
pos = {d: k for k, d in enumerate(ids)}
a = np.array([pos[d] for d in C.d_x]); b = np.array([pos[d] for d in C.d_y])
C["cos_char"] = np.asarray(X[a].multiply(X[b]).sum(1)).ravel()
C["exact"] = df.text.to_numpy()[C.d_x] == df.text.to_numpy()[C.d_y]
for c in ["model", "source", "setting", "prompt_key", "id"]:
    C[c + "_x"] = df[c].to_numpy()[C.d_x]; C[c + "_y"] = df[c].to_numpy()[C.d_y]
C["same_prompt"] = C.prompt_key_x == C.prompt_key_y

SIBSETS = [{"gemini-3.6-flash", "gemini-3.7-flash"}, {"gemini-3.6-flash", "gemini-3.8-flash"},
           {"gemini-3.7-flash", "gemini-3.8-flash"}, {"glm-5.3", "glm-5.3-flash"},
           {"gpt-5.6-luna", "gpt-5.6-sol"}, {"gpt-5.6-luna", "gpt-5.6-terra"},
           {"gpt-5.6-sol", "gpt-5.6-terra"}, {"claude-opus-4-8", "claude-sonnet-5"}]
fam = lambda m: m.split("-")[0].replace("qwen3.8", "qwen")
def cat(r):
    if r.model_x == r.model_y: return "same-model"
    if {r.model_x, r.model_y} in SIBSETS: return "sibling: " + " | ".join(sorted([r.model_x, r.model_y]))
    if fam(r.model_x) == fam(r.model_y): return "same-family other"
    return "cross-family"
C["category"] = C.apply(cat, axis=1)
C[C.cos_char > 0.80].to_csv(OUT / "h1_neardup_pairs.csv", index=False, float_format="%.4f")
cnt = (C.assign(gt095=C.cos_char > 0.95, gt090=C.cos_char > 0.90)
         .groupby("category").agg(n_gt090=("gt090", "sum"), n_gt095=("gt095", "sum"),
                                   n_exact=("exact", "sum"),
                                   n_gt095_same_prompt=("gt095", lambda s: int((s & C.loc[s.index, "same_prompt"]).sum())),
                                   max_cos=("cos_char", "max")).reset_index())
cnt.to_csv(OUT / "h1_neardup_counts.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(cnt.to_string(index=False))
print(C.cos_char.describe())
print(f"done {time.time()-t0:.0f}s")
