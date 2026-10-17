"""Compact tables of the step-4 results for the report. Writes out/summary_assign.md"""
import numpy as np
import pandas as pd

from common import OUT

KEY = ["center_b0.5", "repulse_a0.2", "center_b1.0", "hungarian", "distinct_marg_t1", "center0.5+hungarian",
       "center0.5+sinkhorn0.5+distinct_t1", "sinkhorn_a0.5", "sinkhorn_a1", "count_balance",
       "twin_add_w1.0", "twin_add1+hungarian", "twin_add1+sinkhorn0.5+distinct", "twin_replace"]
SCENS = ["full18", "drop0-30", "uneq_keep", "n3-8", "n1-5", "n2-3", "n1-2", "dup_pairs", "dup_n2-4"]
lines = []
import re
from collections import defaultdict
files = defaultdict(list)
for f in sorted(OUT.glob("assign_*_thr*.csv")):
    files[re.sub(r"_part\w+$", "", f.stem)].append(f)
for stem, fl in sorted(files.items()):
    R = pd.concat([pd.read_csv(x) for x in fl], ignore_index=True)
    class F: pass
    f = F(); f.stem = stem
    base = R[R.decoder == "base"].set_index(["scenario", "seed", "grouping"]).bacc
    R["delta"] = R.bacc.values - base.loc[list(zip(R.scenario, R.seed, R.grouping))].values
    lines.append(f"\n#### {f.stem}\n")
    for grouping in [g for g in ["true", "rec", "rec_dis", "rec_k18"] if g in set(R.grouping)]:
        sub = R[R.grouping == grouping]
        sc = [s for s in SCENS if s in set(sub.scenario)]
        m = sub.groupby(["decoder", "scenario"]).delta.mean().unstack()
        se = sub.groupby(["decoder", "scenario"]).delta.std().unstack() / np.sqrt(sub.groupby(["decoder", "scenario"]).delta.count().unstack())
        b = sub[sub.decoder == "base"].groupby("scenario").bacc.mean()
        lines.append(f"\ngrouping **{grouping}** (delta balanced accuracy vs independent argmax; +-1 s.e. over seeds)\n")
        lines.append("| decoder | " + " | ".join(sc) + " |")
        lines.append("|---|" + "---|" * len(sc))
        lines.append("| *base (argmax) bacc* | " + " | ".join(f"{b[s]:.3f}" for s in sc) + " |")
        for d in KEY:
            if d not in m.index:
                continue
            cells = []
            for s in sc:
                v, e = m.loc[d, s], se.loc[d, s]
                cells.append(f"{v:+.3f}" + ("" if np.isnan(e) else f" ({e:.3f})"))
            lines.append(f"| {d} | " + " | ".join(cells) + " |")
        # best decoder per scenario
        best = m.drop(index="base").idxmax()
        lines.append("| *best decoder* | " + " | ".join(f"{best[s]} {m.loc[best[s], s]:+.3f}" for s in sc) + " |")
(OUT / "summary_assign.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
