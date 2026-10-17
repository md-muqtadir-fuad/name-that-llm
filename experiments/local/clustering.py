"""Step 3: recover prompt groups from passage content only. Writes out/clustering.md / .csv

Pools are simulated from train (150 prompts) with the scenario samplers in structure.py.
'+distractors' pools add passages from 1000 *other* prompts (seyitkaangunes/ntl-extra-api-passages,
Claude-only, generated with a protocol mimicking the competition) so that the number of prompts in the pool
is closer to a real ~4000-passage test (222 prompts if full blocks, ~1300-2000 if 2-3 per prompt).
"""
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from common import OUT, ROOT, load_train
from structure import agglo, agglo_multi, cap_clusters, cluster_metrics, content_tfidf, cosine_dist, sample_pool

df = load_train()
g_all, y_all, texts_all = df.g.values, df.y.values, df.text.values
ext = pd.read_csv(ROOT / "research/ext_data/extra_api/ntl_extra_passages_claude.csv")
ext["g"] = ext.prompt_id.astype("category").cat.codes + 10000
ext_ids = ext.g.unique()
print("extra passages", ext.shape, "prompts", len(ext_ids))

N_PER = {"n3-8": (3, 5), "n1-5": (1, 5), "n2-3": (2, 3), "n1-2": (1, 2)}


def build_pool(scen, seed, distract):
    rng = np.random.default_rng(seed)
    idx = sample_pool(g_all, y_all, scen, rng)
    texts = list(texts_all[idx])
    groups = list(g_all[idx])
    labels = list(y_all[idx])
    if distract:
        if scen in N_PER:
            pids = ext_ids
            lo, hi = N_PER[scen]
        else:
            pids = rng.choice(ext_ids, 100, replace=False)
            lo, hi = 5, 5
        for p in pids:
            rows = np.flatnonzero(ext.g.values == p)
            n = min(rng.integers(lo, hi + 1), len(rows))
            pick = rng.choice(rows, n, replace=False)
            texts += list(ext.text.values[pick])
            groups += [p] * n
            labels += [-1 - i for i in range(n)]  # unique dummy labels (unknown models within Claude)
    return idx, np.array(texts, dtype=object), np.array(groups), np.array(labels)


METHODS = {
    "tfidf_uni": dict(ngram=(1, 1), lsa=None),
    "tfidf_uni_bi": dict(ngram=(1, 2), lsa=None),
    "tfidf_lsa128": dict(ngram=(1, 1), lsa=128),
}
THRS = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

rows = []
for distract in (False, True):
    for scen in ["full18", "drop0-30", "uneq_keep", "n3-8", "n1-5", "n2-3", "n1-2"]:
        for seed in range(2):
            idx, texts, groups, labels = build_pool(scen, 100 + seed, distract)
            for mname, mcfg in METHODS.items():
                t = time.time()
                X = content_tfidf(texts, ngram=mcfg["ngram"])
                if mcfg["lsa"]:
                    X = normalize(TruncatedSVD(mcfg["lsa"], random_state=0).fit_transform(X))
                D = cosine_dist(X).astype(np.float64)
                thrs = THRS if not mcfg["lsa"] else [t - 0.4 for t in THRS if t <= 0.8]  # LSA distances are smaller
                cuts = agglo_multi(D, thrs)
                for thr_eff in thrs:
                    lab = cap_clusters(cuts[thr_eff], D, cap=18)
                    m = cluster_metrics(groups, lab, labels)
                    m.update(scenario=scen, distract=distract, seed=seed, method=mname, thr=thr_eff,
                             n_pool=len(texts), n_prompts=len(np.unique(groups)))
                    rows.append(m)
                print(scen, distract, seed, mname, len(texts), f"{time.time()-t:.1f}s", flush=True)

R = pd.DataFrame(rows)
R.to_csv(OUT / "clustering.csv", index=False)
agg = (R.groupby(["distract", "scenario", "method", "thr"])
       [["ari", "purity", "pair_prec", "pair_rec", "singleton_frac", "mean_size", "dup_label_clusters", "n_pool", "n_prompts"]]
       .mean().reset_index())
lines = ["## 3. Prompt-group recovery (content-only clustering)\n",
         "Average-linkage agglomerative clustering on cosine distance; TF-IDF (english stop words removed, sublinear) "
         "fit on the pool only; clusters > 18 split. Means over 2 seeds.\n"]
for distract in (False, True):
    lines.append(f"\n### Pools {'WITH 1000-prompt distractors (realistic size)' if distract else 'from train only (150 prompts)'}\n")
    best = agg[agg.distract == distract].sort_values("ari", ascending=False).groupby(["scenario", "method"]).head(1)
    lines.append("| scenario | method | best thr | ARI | purity | pair P | pair R | singletons | mean cluster size | clusters w/ duplicate label | pool size | #prompts |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in best.sort_values(["scenario", "method"]).iterrows():
        lines.append(f"| {r.scenario} | {r.method} | {r.thr:.2f} | {r.ari:.3f} | {r.purity:.3f} | {r.pair_prec:.3f} | {r.pair_rec:.3f} | "
                     f"{r.singleton_frac:.2f} | {r.mean_size:.1f} | {r.dup_label_clusters:.3f} | {r.n_pool:.0f} | {r.n_prompts:.0f} |")
    lines.append("\nThreshold sweep, tfidf_uni (ARI / pair-precision / pair-recall):\n")
    sub = agg[(agg.distract == distract) & (agg.method == "tfidf_uni")]
    piv = sub.pivot(index="scenario", columns="thr", values=["ari", "pair_prec", "pair_rec"])
    lines.append("| scenario | " + " | ".join(f"thr {t:.2f}" for t in sorted(sub.thr.unique())) + " |")
    lines.append("|---|" + "---|" * sub.thr.nunique())
    for s in piv.index:
        lines.append(f"| {s} | " + " | ".join(
            f"{piv.loc[s, ('ari', t)]:.2f}/{piv.loc[s, ('pair_prec', t)]:.2f}/{piv.loc[s, ('pair_rec', t)]:.2f}"
            for t in sorted(sub.thr.unique())) + " |")
(OUT / "clustering.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
