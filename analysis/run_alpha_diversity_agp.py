"""
AGP alpha-diversity analysis: Breakaway (richness) + DivNet (Shannon).
Replicates the KORA analysis in notebooks/02_richness.ipynb and
notebooks/03_alpha-diversity.ipynb on the matched AGP smoking dataset.

Inputs (relative to repo root):
  source/design_AG/otu_table_smoker.csv      -- 234 x 40 count table (samples x OTUs)
  source/design_AG/otu_table_non_smoker.csv  -- 234 x 40 count table
  source/design_AG/sample_data_smoker.csv    -- matched IDs + pair_nb
  source/design_AG/sample_data_non_smoker.csv

Outputs (results/agp/alpha_diversity/):
  richness_betta_table.csv     -- betta regression on breakaway estimates
  divnet_shannon.csv           -- DivNet Shannon estimates per sample
  divnet_betta_table.csv       -- betta regression on Shannon estimates
  richness_agp_smoking.png/svg -- boxplot: richness by smoking status
  alpha_diversity_agp_smoking.png/svg -- boxplot: Shannon by smoking status
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
import rpy2.robjects.pandas2ri as pandas2ri

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(REPO_ROOT, "source", "design_AG")
OUT_DIR   = os.path.join(REPO_ROOT, "results", "agp", "alpha_diversity")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load R packages ───────────────────────────────────────────────────────────
base      = importr("base")
utils     = importr("utils")
breakaway = importr("breakaway")
divnet    = importr("DivNet")

# ── Load data ─────────────────────────────────────────────────────────────────
otu_sm  = pd.read_csv(os.path.join(DATA_DIR, "otu_table_smoker.csv"),     index_col=0)
otu_ns  = pd.read_csv(os.path.join(DATA_DIR, "otu_table_non_smoker.csv"), index_col=0)
meta_sm = pd.read_csv(os.path.join(DATA_DIR, "sample_data_smoker.csv"),   index_col=0)
meta_ns = pd.read_csv(os.path.join(DATA_DIR, "sample_data_non_smoker.csv"), index_col=0)
tax_sm  = pd.read_csv(os.path.join(DATA_DIR, "tax_table_smoker.csv"),     index_col=0)

family_map = {str(idx): (fam if pd.notna(fam) else f"f__unknown_{idx}")
              for idx, fam in tax_sm["Family"].items()}

# Smoking indicator: 1 = smoker, 0 = never-smoker
meta_sm["W"]     = 1
meta_sm["W_str"] = "Smoker"
meta_ns["W"]     = 0
meta_ns["W_str"] = "Never-smoker"

otu  = pd.concat([otu_sm,  otu_ns],  axis=0)
meta = pd.concat([meta_sm, meta_ns], axis=0)
meta = meta.loc[otu.index]          # align rows

print(f"Combined OTU table: {otu.shape}  (samples x OTUs)")
print(f"Combined metadata:  {meta.shape}")

# Transpose: OTUs (rows) x samples (columns) — required by breakaway/DivNet
otu_T = otu.T.astype(int)
# Rename rows from numeric OTU IDs to family names (DivNet requires named rows)
otu_T.index = [family_map.get(str(c), str(c)) for c in otu_T.index]
print(f"Transposed OTU table: {otu_T.shape}  (OTUs x samples)")

# ── Richness via Breakaway + betta ────────────────────────────────────────────
print("\n=== Breakaway richness estimation ===")
with localconverter(ro.default_converter + pandas2ri.converter):
    r_otu = ro.conversion.py2rpy(otu_T)
ba    = breakaway.breakaway(r_otu)
summ  = base.summary(ba)
sum_dict = dict(zip(summ.names, map(list, summ)))

estimates = np.round(sum_dict["estimate"], 4)
errors    = np.round(sum_dict["error"],    4)

W          = meta["W"].reset_index(drop=True)
design_mat = pd.DataFrame({"intercept": 1, "W": W})

with localconverter(ro.default_converter + pandas2ri.converter):
    betta_res = breakaway.betta(chats=FloatVector(estimates),
                                ses=FloatVector(errors),
                                X=design_mat)
betta_table = pd.DataFrame(betta_res[0],
                            columns=["estimate", "error", "p_value"],
                            index=design_mat.columns)
print(betta_table)
betta_table.to_csv(os.path.join(OUT_DIR, "richness_betta_table.csv"))

# Richness boxplot
rich_df = pd.DataFrame(sum_dict)
rich_df.index = rich_df["sample_names"].astype(str)
rich_df = rich_df.join(meta[["W_str"]])
rich_df.rename(columns={"W_str": "Smoking"}, inplace=True)

vc  = rich_df["Smoking"].value_counts()
fig = px.box(rich_df, x="Smoking", y="estimate", color="Smoking",
             color_discrete_map={"Smoker": "red", "Never-smoker": "green"},
             width=500, height=800)
anns = [go.layout.Annotation(x=k, y=rich_df[rich_df["Smoking"]==k]["estimate"].max()+5,
                               text=str(v), showarrow=False, font=dict(size=12))
        for k, v in vc.items()]
fig.update_layout(title="Richness box-plot AGP (smoking)", annotations=anns)
fig.write_image(os.path.join(OUT_DIR, "richness_agp_smoking.png"))
fig.write_image(os.path.join(OUT_DIR, "richness_agp_smoking.svg"))
print(f"Richness plot saved.")

# ── Shannon via DivNet + betta ────────────────────────────────────────────────
print("\n=== DivNet Shannon estimation ===")

# Base taxon: most prevalent family (highest total count across samples)
base_taxon = str(otu_T.sum(axis=1).idxmax())
print(f"Base taxon for DivNet: {base_taxon}")

with localconverter(ro.default_converter + pandas2ri.converter):
    r_otu_div = ro.conversion.py2rpy(otu_T)
dv = divnet.divnet(r_otu_div, base=base_taxon, ncores=4)

shannon = dv[0]
div_dict = {}
for i in range(len(shannon)):
    sid = str(shannon.names[i])
    div_dict[sid] = (round(float(shannon[i][0]), 4),
                     round(float(shannon[i][1]), 4))

div_df = pd.DataFrame.from_dict(div_dict, orient="index", columns=["estimate", "error"])
div_df.to_csv(os.path.join(OUT_DIR, "divnet_shannon.csv"))

# Align metadata to DivNet output order
meta_aligned = meta.loc[div_df.index] if all(i in meta.index for i in div_df.index) else meta
W_div        = meta_aligned["W"].reset_index(drop=True)
design_div   = pd.DataFrame({"intercept": 1, "W": W_div})

with localconverter(ro.default_converter + pandas2ri.converter):
    betta_div = breakaway.betta(chats=FloatVector(div_df["estimate"].values),
                                ses=FloatVector(div_df["error"].values),
                                X=design_div)
betta_div_table = pd.DataFrame(betta_div[0],
                                columns=["estimate", "error", "p_value"],
                                index=design_div.columns)
print(betta_div_table)
betta_div_table.to_csv(os.path.join(OUT_DIR, "divnet_betta_table.csv"))

# Shannon boxplot
div_plot = div_df.join(meta_aligned[["W_str"]])
div_plot.rename(columns={"W_str": "Smoking"}, inplace=True)

vc2  = div_plot["Smoking"].value_counts()
fig2 = px.box(div_plot, x="Smoking", y="estimate", color="Smoking",
              color_discrete_map={"Smoker": "red", "Never-smoker": "green"},
              width=500, height=800)
anns2 = [go.layout.Annotation(x=k,
                                y=div_plot[div_plot["Smoking"]==k]["estimate"].max()+0.05,
                                text=str(v), showarrow=False, font=dict(size=12))
         for k, v in vc2.items()]
fig2.update_layout(title="Alpha diversity (Shannon) box-plot AGP (smoking)", annotations=anns2)
fig2.write_image(os.path.join(OUT_DIR, "alpha_diversity_agp_smoking.png"))
fig2.write_image(os.path.join(OUT_DIR, "alpha_diversity_agp_smoking.svg"))
print(f"Shannon plot saved.")

print(f"\n=== Done. Outputs in {OUT_DIR} ===")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
