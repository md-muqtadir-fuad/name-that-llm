"""H1 step 1: same-prompt pairwise similarities for every pair of passages that share a prompt.

Inputs (read-only): data/train.csv, research/ext_data/twostage/pool_v4.parquet
Output: research/bottleneck/h1_near_identity/pairs.parquet (one row per same-prompt passage pair)
        research/bottleneck/h1_near_identity/docs_meta.parquet (per-passage metadata)
Metrics per pair:
  cos_char   char_wb 3-5-gram TF-IDF cosine (hashed 2^21, sublinear tf, idf fit on pool+train)
  cos_word   word 1-2-gram TF-IDF cosine
  len_ratio  min(n_words)/max(n_words)
  sent_diff  |#sentences_i - #sentences_j|
  ident_sent share of identical normalised sentences (>=4 words), |A&B|/min(|A|,|B|)
"""
import re, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
t0 = time.time()

pool = pd.read_parquet(ROOT / "research/ext_data/twostage/pool_v4.parquet")
tr = pd.read_csv(ROOT / "data/train.csv")
tr = tr.assign(source="train", setting="train", n_words=tr.text.str.split().str.len(),
               prompt_key="train:" + tr.prompt_id.astype(str))
df = pd.concat([pool[["id", "prompt_key", "text", "model", "source", "setting", "n_words"]],
                tr[["id", "prompt_key", "text", "model", "source", "setting", "n_words"]]],
               ignore_index=True)
print("docs", len(df), f"{time.time()-t0:.0f}s", flush=True)

# --- vectors
hv_c = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**21,
                         alternate_sign=False, norm=None, lowercase=True)
Xc = TfidfTransformer(sublinear_tf=True).fit_transform(hv_c.transform(df.text)).tocsr()
print("char tfidf", Xc.shape, Xc.nnz, f"{time.time()-t0:.0f}s", flush=True)
hv_w = HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=2**20,
                         alternate_sign=False, norm=None, lowercase=True)
Xw = TfidfTransformer(sublinear_tf=True).fit_transform(hv_w.transform(df.text)).tocsr()
print("word tfidf", f"{time.time()-t0:.0f}s", flush=True)

# --- sentences
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
def norm_sents(t):
    ss = [s.strip() for s in SPLIT.split(t) if s.strip()]
    keys = set()
    for s in ss:
        k = re.sub(r"[^\w\s]", "", s.lower())
        k = re.sub(r"\s+", " ", k).strip()
        if len(k.split()) >= 4:
            keys.add(k)
    return len(ss), frozenset(keys)
sent = [norm_sents(t) for t in df.text]
nsent = np.array([s[0] for s in sent])
skeys = [s[1] for s in sent]
nw = df.n_words.to_numpy()

# --- pairs within prompt_key
rows = []
for pk, idx in df.groupby("prompt_key").indices.items():
    if len(idx) < 2:
        continue
    Cc = (Xc[idx] @ Xc[idx].T).toarray()
    Cw = (Xw[idx] @ Xw[idx].T).toarray()
    a, b = np.triu_indices(len(idx), 1)
    I, J = idx[a], idx[b]
    rows.append(pd.DataFrame({"i": I, "j": J, "cos_char": Cc[a, b].astype(np.float32),
                              "cos_word": Cw[a, b].astype(np.float32)}))
P = pd.concat(rows, ignore_index=True)
print("pairs", len(P), f"{time.time()-t0:.0f}s", flush=True)
I, J = P.i.to_numpy(), P.j.to_numpy()
P["len_ratio"] = (np.minimum(nw[I], nw[J]) / np.maximum(nw[I], nw[J])).astype(np.float32)
P["sent_diff"] = np.abs(nsent[I] - nsent[J]).astype(np.int16)
P["ident_sent"] = np.array([len(skeys[i] & skeys[j]) / min(len(skeys[i]), len(skeys[j]))
                            if skeys[i] and skeys[j] else 0.0 for i, j in zip(I, J)], dtype=np.float32)
for c in ["prompt_key", "model", "source", "setting"]:
    P[c + "_i"] = df[c].to_numpy()[I]
    if c != "prompt_key":
        P[c + "_j"] = df[c].to_numpy()[J]
P = P.rename(columns={"prompt_key_i": "prompt_key"})
for c in P.columns:
    if P[c].dtype == object:
        P[c] = P[c].astype("category")
P.to_parquet(OUT / "pairs.parquet", index=False)
df.drop(columns="text").assign(n_sent=nsent).to_parquet(OUT / "docs_meta.parquet", index=False)
print("saved", f"{time.time()-t0:.0f}s", flush=True)
