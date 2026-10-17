"""Step 4: value of prompt-structure decoding on OOF predictions.

Score sets:
  tfidf  - our TF-IDF(char_wb 1-5 + word 1-2) + stylometry LinearSVC (plain) and its chunk-centred twin (twin.py)
  strong - public frozen-LLM plain blend (lightningv08/ntl-oof-q35-heads, prompt-disjoint OOF, plain members only)
Groupings: true prompt ids ("oracle") or recovered by content TF-IDF clustering of the pool (+ optional distractors).
Decoders: centering, repulsion, Hungarian, exact distinct-author marginals, twin add-mode, global balancing.
Writes out/assign_<set>.csv and out/assign_<set>.md
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from common import OUT, ROOT, K, load_train, bacc, log_softmax
from structure import (agglo, cap_clusters, center, content_tfidf, cosine_dist, exact_distinct, hungarian,
                       repulsion, sample_pool, sinkhorn_global, bias_balance_counts, groups_iter)

SET = sys.argv[1] if len(sys.argv) > 1 else "tfidf"
GROUPINGS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["true", "rec", "rec_dis"]
THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.80
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 5

df = load_train()
y, g, texts = df.y.values, df.g.values, df.text.values
ext = pd.read_csv(ROOT / "research/ext_data/extra_api/ntl_extra_passages_claude.csv")
ext_g = ext.prompt_id.astype("category").cat.codes.values + 10000
ext_ids = np.unique(ext_g)

if SET == "tfidf":
    z = np.load(OUT / "twin_scores_char_wb.npz")
    Tp, Tt = float(z["Tp"]), float(z["Tt"])
    L_all = log_softmax(z["S"] * Tp)
    TW_all = z["T"]; TB_all = z["B"]
    L_ext = log_softmax(z["S_ext"] * Tp)
    TW_ext = z["T_ext"]; TB_ext = z["B_ext"]
else:
    zz = np.load(ROOT / "research/ext_oof/q35heads/oof_predictions.npz", allow_pickle=False)
    assert (zz["ids"] == df.id.values).all()
    names = list(zz["component_names"])
    w = json.loads((ROOT / "research/ext_oof/q35heads/bundle_manifest.json").read_text())["weights"]
    wp = {k: v for k, v in w.items() if "center" not in k}  # plain members only (no centring inside)
    L_all = sum(v * zz["components"][names.index(k), 0] for k, v in wp.items()) / sum(wp.values())
    # temperature to make it a calibrated log-prob
    from scipy.optimize import minimize_scalar
    f = lambda t: -log_softmax(L_all * t)[np.arange(len(y)), y].mean()
    t = minimize_scalar(f, bounds=(0.1, 20), method="bounded").x
    L_all = log_softmax(L_all * t)
    TW_all = None
    print(f"strong plain blend OOF bacc {bacc(y, L_all.argmax(1)):.4f}, T={t:.2f}")

N_PER = {"n3-8": (3, 5), "n1-5": (1, 5), "n2-3": (2, 3), "n1-2": (1, 2)}


def twin_logp(TW, TB, groups):
    """Twin head on group-centred features: raw - group mean(raw) + b; singletons -> None mask."""
    Z = np.zeros_like(TW)
    ok = np.zeros(len(TW), bool)
    for m in groups_iter(groups):
        if len(m) >= 2:
            Z[m] = TW[m] - TW[m].mean(0, keepdims=True) + TB[m]
            ok[m] = True
    return log_softmax(Z * Tt), ok


def decoders(L, groups, TW=None, TB=None):
    out = {"base": L.argmax(1)}
    for b in (0.25, 0.5, 0.75, 1.0):
        out[f"center_b{b}"] = center(L, groups, b).argmax(1)
    for a in (0.2, 0.35, 0.5, 0.7):
        out[f"repulse_a{a}"] = repulsion(L, groups, a).argmax(1)
    out["hungarian"] = hungarian(L, groups)
    out["center1+hungarian"] = hungarian(center(L, groups, 1.0), groups)
    C5 = center(L, groups, 0.5)
    out["center0.5+hungarian"] = hungarian(C5, groups)
    out["center0.5+distinct_t1"] = exact_distinct(C5, groups, 1.0, max_n=10).argmax(1)
    out["center0.5+sinkhorn0.5"] = sinkhorn_global(C5, 0.5).argmax(1)
    out["center0.5+sinkhorn0.5+distinct_t1"] = exact_distinct(sinkhorn_global(C5, 0.5), groups, 1.0, max_n=10).argmax(1)
    out["distinct_marg_t1"] = exact_distinct(L, groups, 1.0, max_n=10).argmax(1)
    out["distinct_marg_t0.5"] = exact_distinct(L, groups, 0.5, max_n=10).argmax(1)
    out["sinkhorn_a0.5"] = sinkhorn_global(L, 0.5).argmax(1)
    out["sinkhorn_a1"] = sinkhorn_global(L, 1.0).argmax(1)
    out["count_balance"] = bias_balance_counts(L).argmax(1)
    out["center1+sinkhorn0.5"] = sinkhorn_global(center(L, groups, 1.0), 0.5).argmax(1)
    out["sinkhorn0.5+hungarian"] = hungarian(sinkhorn_global(L, 0.5), groups)
    out["sinkhorn0.5+distinct_t1"] = exact_distinct(sinkhorn_global(L, 0.5), groups, 1.0, max_n=10).argmax(1)
    if TW is not None:
        LT, ok = twin_logp(TW, TB, groups)
        for wt in (0.5, 1.0, 2.0):
            Z = L.copy()
            Z[ok] = L[ok] + wt * LT[ok]
            out[f"twin_add_w{wt}"] = Z.argmax(1)
            if wt == 1.0:
                out["twin_add1+hungarian"] = hungarian(Z, groups)
                out["twin_add1+distinct_t1"] = exact_distinct(Z, groups, 1.0, max_n=10).argmax(1)
                out["twin_add1+sinkhorn0.5"] = sinkhorn_global(Z, 0.5).argmax(1)
                out["twin_add1+sinkhorn0.5+distinct"] = exact_distinct(sinkhorn_global(Z, 0.5), groups, 1.0, max_n=10).argmax(1)
        Z = L.copy(); Z[ok] = LT[ok]
        out["twin_replace"] = Z.argmax(1)
    return out


rows = []
t0 = time.time()
SCENS = ["full18", "drop0-30", "uneq_keep", "n3-8", "n1-5", "n2-3", "n1-2", "dup_pairs", "dup_n2-4"]
import os
if os.environ.get("SCENS"):
    SCENS = os.environ["SCENS"].split(",")
CHUNK = os.environ.get("CHUNK", "")


def sample_dup(rng, lo, hi):
    """Repeated-author stress test: merge random PAIRS of train prompts into one pseudo-prompt
    (each model can then appear twice) and sample lo..hi passages from the 36."""
    prompts = rng.permutation(np.unique(g))
    gid = np.empty(len(g), int)
    for i in range(0, len(prompts) - 1, 2):
        gid[np.isin(g, prompts[i:i + 2])] = i
    if len(prompts) % 2:
        gid[g == prompts[-1]] = len(prompts) - 1
    keep = []
    for q in np.unique(gid):
        r = np.flatnonzero(gid == q)
        keep.extend(rng.choice(r, min(rng.integers(lo, hi + 1), len(r)), replace=False))
    idx = np.sort(np.array(keep))
    return idx, gid[idx]


for scen in SCENS:
    nseeds = 1 if scen == "full18" else (SEEDS if scen in ("drop0-30", "uneq_keep", "dup_pairs") else 2 * SEEDS)
    for seed in range(nseeds):
        rng = np.random.default_rng(500 + seed)
        if scen.startswith("dup"):
            idx, true_groups = sample_dup(rng, *((8, 18) if scen == "dup_pairs" else (2, 4)))
        else:
            idx = sample_pool(g, y, scen, rng)
            true_groups = g[idx]
        for grouping in GROUPINGS:
            if scen.startswith("dup") and grouping != "true":
                continue
            L = L_all[idx]
            TW = TW_all[idx] if TW_all is not None else None
            TB = TB_all[idx] if TW_all is not None else None
            n_train = len(idx)
            if grouping == "true":
                groups = true_groups
            elif grouping == "rec_k18":  # 'assume 18 answers per prompt': cut the tree into N/18 clusters
                X = content_tfidf(np.array(list(texts[idx]), dtype=object))
                D = cosine_dist(X).astype(np.float64)
                from scipy.cluster.hierarchy import linkage, fcluster
                from scipy.spatial.distance import squareform
                Zl = linkage(squareform(D, checks=False), method="average")
                groups = cap_clusters(fcluster(Zl, max(1, round(len(idx) / 18)), criterion="maxclust") - 1, D, 18)
            else:
                ptexts = list(texts[idx])
                if grouping == "rec_dis":
                    if scen in N_PER:
                        pids, (lo, hi) = ext_ids, N_PER[scen]
                    else:
                        pids, (lo, hi) = rng.choice(ext_ids, 100, replace=False), (5, 5)
                    eidx = []
                    for p in pids:
                        r = np.flatnonzero(ext_g == p)
                        eidx.extend(rng.choice(r, min(rng.integers(lo, hi + 1), len(r)), replace=False))
                    eidx = np.array(eidx)
                    ptexts += list(ext.text.values[eidx])
                X = content_tfidf(np.array(ptexts, dtype=object))
                D = cosine_dist(X).astype(np.float64)
                groups = cap_clusters(agglo(D, THR), D, 18)
                if grouping == "rec_dis":
                    if SET == "tfidf":  # distractors take part in decoding with their own scores
                        L = np.vstack([L, L_ext[eidx]])
                        TW = np.vstack([TW, TW_ext[eidx]]); TB = np.vstack([TB, TB_ext[eidx]])
                    else:  # no strong scores for distractors: drop them after clustering
                        groups = groups[:n_train]
            preds = decoders(L, groups, TW, TB)
            sizes = np.bincount(groups)[groups][:n_train]
            for name, p in preds.items():
                rows.append(dict(scenario=scen, seed=seed, grouping=grouping, decoder=name,
                                 bacc=bacc(y[idx], p[:n_train]), n=n_train,
                                 grouped_frac=float(np.mean(sizes >= 2)), mean_size=float(np.mean(sizes))))
        print(scen, seed, f"{time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rows)
tag = f"{SET}_thr{THR:.2f}" + (f"_part{CHUNK}" if CHUNK else "")
R.to_csv(OUT / f"assign_{tag}.csv", index=False)
base = R[R.decoder == "base"].set_index(["scenario", "seed", "grouping"]).bacc
R["delta"] = R.bacc.values - base.loc[list(zip(R.scenario, R.seed, R.grouping))].values
order = SCENS
lines = [f"### 4.x score set `{SET}` (clustering thr {THR}), balanced accuracy, mean over seeds\n"]
for grouping in GROUPINGS:
    sub = R[R.grouping == grouping]
    order = [o for o in SCENS if o in set(sub.scenario)]
    piv = sub.groupby(["decoder", "scenario"]).delta.mean().unstack()[order]
    sd = sub.groupby(["decoder", "scenario"]).delta.std().unstack()[order]
    b = sub[sub.decoder == "base"].groupby("scenario").bacc.mean()[order]
    gf = sub[sub.decoder == "base"].groupby("scenario").grouped_frac.mean()[order]
    lines.append(f"\n**grouping = {grouping}**  (rows: decoder; cells: delta vs base, +-sd over seeds)\n")
    lines.append("| decoder | " + " | ".join(order) + " |")
    lines.append("|---|" + "---|" * len(order))
    lines.append("| base bacc | " + " | ".join(f"{b[s]:.4f}" for s in order) + " |")
    lines.append("| share of rows in groups>=2 | " + " | ".join(f"{gf[s]:.2f}" for s in order) + " |")
    for d in piv.index:
        if d == "base":
            continue
        lines.append(f"| {d} | " + " | ".join(
            f"{piv.loc[d, s]:+.4f}" + (f" +-{sd.loc[d, s]:.3f}" if not np.isnan(sd.loc[d, s]) else "") for s in order) + " |")
(OUT / f"assign_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
