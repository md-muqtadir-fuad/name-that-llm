"""H1 step 5: one-row-per-pair verdict table from the step outputs. Writes h1_verdict.csv."""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent
s = pd.read_csv(OUT / "h1_pair_summary.csv")
s = s[s.metric == "cos_char"].set_index("pair")
sp = pd.read_csv(OUT / "h1_samemodel_split.csv").set_index("pair")
c = pd.read_csv(OUT / "h1_pairclf.csv").set_index("contrast")
cp = pd.read_csv(OUT / "h1_pairclf_power3000.csv").set_index("contrast")
nd = pd.read_csv(OUT / "h1_neardup_counts.csv")
nd = nd[nd.category.str.startswith("sibling")].assign(pair=lambda d: d.category.str[9:]).set_index("pair")

VERDICT = {
    "gemini-3.7-flash | gemini-3.8-flash": "SUPPORTED: as similar as same-model; residual signal faint",
    "glm-5.3 | glm-5.3-flash": "PARTLY: classifier near chance, similarity only halfway (low dynamic range)",
    "gemini-3.6-flash | gemini-3.7-flash": "NOT SUPPORTED: close in similarity but clearly learnable",
    "gemini-3.6-flash | gemini-3.8-flash": "NOT SUPPORTED: close in similarity but clearly learnable",
    "gpt-5.6-luna | gpt-5.6-sol": "NOT SUPPORTED: learnable",
    "gpt-5.6-luna | gpt-5.6-terra": "NOT SUPPORTED: learnable",
    "gpt-5.6-sol | gpt-5.6-terra": "NOT SUPPORTED: learnable (weakest of the gpt trio)",
    "claude-opus-4-8 | claude-sonnet-5": "NOT SUPPORTED: ordinary within-family distance (proxy baseline)",
}
rows = []
for p in s[s.kind == "sibling"].index:
    rows.append(dict(pair=p, med_S=s.at[p, "med_S"], med_M=s.at[p, "med_M"], med_C=s.at[p, "med_C"],
                     m_source=s.at[p, "m_source"], n_prompts=s.at[p, "n_prompts_matched"],
                     R=s.at[p, "R"], R_lo=s.at[p, "R_lo"], R_hi=s.at[p, "R_hi"],
                     R_condmatched=sp.at[p, "R_xcond"], auc_M_gt_S=s.at[p, "auc_M_gt_S"], dz=s.at[p, "dz"],
                     clf_paired_acc_800=c.at[p, "paired_acc"], clf_single_acc_800=c.at[p, "oof_acc"],
                     clf_paired_acc_power=cp.at[p, "paired_acc"] if p in cp.index else None,
                     clf_power_n=cp.at[p, "n_prompts"] if p in cp.index else None,
                     neardup_gt095=nd.at[p, "n_gt095"] if p in nd.index else 0,
                     max_cos_any_prompt=nd.at[p, "max_cos"] if p in nd.index else None,
                     verdict=VERDICT[p]))
v = pd.DataFrame(rows).sort_values("R", ascending=False)
v.to_csv(OUT / "h1_verdict.csv", index=False, float_format="%.3f")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(v.round(3).to_string(index=False))
print("\ncontrols (classifier):")
print(c[c.kind != "sibling"][["n_prompts", "oof_acc", "paired_acc"]].round(3).to_string())
