"""
AGP differential abundance analysis using LinDA with BH and Lee et al. correction.
Replicates the KORA approach (notebooks/04_LinDA.ipynb) on the matched AGP smoking
dataset at family level.

Inputs (relative to repo root):
  source/design_AG/otu_table_smoker.csv       -- 234 x 40 counts (samples x OTUs)
  source/design_AG/otu_table_non_smoker.csv   -- 234 x 40 counts
  source/design_AG/sample_data_smoker.csv     -- pair_nb, W
  source/design_AG/sample_data_non_smoker.csv
  source/design_AG/tax_table_smoker.csv       -- OTU -> Family taxonomy

Outputs (results/agp/da/):
  linda_agp_results.csv       -- LFC, SE, stat, BH q-value per family
  linda_agp_permstats.csv     -- permutation test statistics (n_iter x p)
  linda_agp_adjpvalues.csv    -- BH and Lee et al. adjusted p-values
  da_family_agp_smoking.png/svg -- effect-size figure (BH / Lee et al. panels)
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.pandas2ri as pandas2ri

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from utils.helper import (calculate_unadjusted_p_values,
                           min_unadjusted_p_values,
                           adjusted_p_values)

def r_to_pandas(r_dataframe):
    """Convert R DataFrame to pandas — inline to avoid helper.py import issue."""
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_dataframe)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(REPO_ROOT, "source", "design_AG")
OUT_DIR  = os.path.join(REPO_ROOT, "results", "agp", "da")
os.makedirs(OUT_DIR, exist_ok=True)

N_PERM = 1000   # permutations for Lee et al. correction
ALPHA  = 0.1    # FDR threshold

# ── R packages ────────────────────────────────────────────────────────────────
linda = importr("LinDA")

# ── Load data ─────────────────────────────────────────────────────────────────
otu_sm  = pd.read_csv(os.path.join(DATA_DIR, "otu_table_smoker.csv"),     index_col=0)
otu_ns  = pd.read_csv(os.path.join(DATA_DIR, "otu_table_non_smoker.csv"), index_col=0)
meta_sm = pd.read_csv(os.path.join(DATA_DIR, "sample_data_smoker.csv"),   index_col=0)
meta_ns = pd.read_csv(os.path.join(DATA_DIR, "sample_data_non_smoker.csv"), index_col=0)
tax_sm  = pd.read_csv(os.path.join(DATA_DIR, "tax_table_smoker.csv"),     index_col=0)

# Family name mapping (OTU numeric ID -> Family string)
family_map = {str(idx): (fam if pd.notna(fam) else f"f__unknown_{idx}")
              for idx, fam in tax_sm["Family"].items()}

# Smoking indicator: 1 = smoker, 0 = never-smoker
meta_sm["W"] = 1
meta_ns["W"] = 0

# Combined count table (samples x OTUs) and metadata
otu  = pd.concat([otu_sm, otu_ns], axis=0)
meta = pd.concat([meta_sm[["pair_nb", "W"]], meta_ns[["pair_nb", "W"]]], axis=0)
meta = meta.loc[otu.index]

print(f"OTU table: {otu.shape}  (samples x OTUs)")
print(f"Metadata:  {meta.shape}")

# Transpose to features x samples for LinDA
otu_T = otu.T.astype(int)
otu_T.index = otu_T.index.map(lambda x: family_map.get(str(x), str(x)))
print(f"Transposed OTU: {otu_T.shape}  (OTUs x samples)")

# Treatment indicator column (samples x 1)
w_obs = meta[["W"]].copy()
w_obs.columns = ["w"]

# ── LinDA on observed assignment ──────────────────────────────────────────────
print("\n=== LinDA: observed assignment ===")
with localconverter(ro.default_converter + pandas2ri.converter):
    r_otu = ro.conversion.py2rpy(otu_T)
    r_w   = ro.conversion.py2rpy(w_obs)
# Explicitly set row names so LinDA output carries family names as index
r_otu.rownames = ro.StrVector(otu_T.index.tolist())
print(f"r_otu rownames (first 3): {list(r_otu.rownames)[:3]}")

lo   = linda.linda(r_otu, r_w, formula="~w", alpha=ALPHA, prev_cut=0.0, lib_cut=1)
out  = r_to_pandas(lo.rx2("output").rx2("w"))

results = pd.DataFrame({
    "family":         out.index,
    "lfc":            out["log2FoldChange"].values,
    "lfcSE":          out["lfcSE"].values,
    "stat":           out["stat"].values,
    "pvalue":         out["pvalue"].values,
    "padj_bh":        out["padj"].values,
    "reject_bh":      out["reject"].values,
}).set_index("family")

results.to_csv(os.path.join(OUT_DIR, "linda_agp_results.csv"))
print(f"Significant (BH q<{ALPHA}): {results['reject_bh'].sum()} of {len(results)} families")
print(results[results["reject_bh"]].sort_values("padj_bh")[["lfc", "pvalue", "padj_bh"]])

# ── Lee et al. permutation correction ────────────────────────────────────────
print(f"\n=== Lee et al. correction ({N_PERM} within-pair permutations) ===")

pair_ids  = meta["pair_nb"].unique()
perm_stats = np.empty((len(results), 0))

for i in range(N_PERM):
    if (i + 1) % 100 == 0:
        print(f"  permutation {i+1}/{N_PERM}")

    # Flip labels within each matched pair independently
    w_perm = meta[["W", "pair_nb"]].copy()
    for pid in pair_ids:
        mask = w_perm["pair_nb"] == pid
        if np.random.rand() > 0.5:
            w_perm.loc[mask, "W"] = 1 - w_perm.loc[mask, "W"]
    w_perm = w_perm[["W"]].rename(columns={"W": "w"})

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_w_perm = ro.conversion.py2rpy(w_perm)
    lo_p     = linda.linda(r_otu, r_w_perm, formula="~w",
                           alpha=ALPHA, prev_cut=0.0, lib_cut=1)
    out_p    = r_to_pandas(lo_p.rx2("output").rx2("w"))
    perm_stats = np.hstack([perm_stats,
                             out_p["stat"].reindex(results.index).values.reshape(-1, 1)])

perm_df = pd.DataFrame(perm_stats, index=results.index)
perm_df.to_csv(os.path.join(OUT_DIR, "linda_agp_permstats.csv"))

# Lee et al. p-values
obs_stat   = results["stat"]
pval_lee   = calculate_unadjusted_p_values(perm_df, obs_stat, test_type="two-sided")
min_pvals  = min_unadjusted_p_values(perm_df)
# adjusted_p_values uses integer indexing internally; reset index before passing
adj_pvals  = adjusted_p_values(min_pvals, pval_lee.reset_index(drop=True))

adj_df = pval_lee.copy()
adj_df.columns = ["p_perm"]
adj_df["q_lee"] = adj_pvals
adj_df["reject_lee"] = adj_df["q_lee"] < ALPHA

results = results.join(adj_df)
results.to_csv(os.path.join(OUT_DIR, "linda_agp_adjpvalues.csv"))

n_bh  = results["reject_bh"].sum()
n_lee = results["reject_lee"].sum()
print(f"Significant (BH q<{ALPHA}):       {n_bh} families")
print(f"Significant (Lee et al. q<{ALPHA}): {n_lee} families")

# ── Plot: two-panel effect-size figure (BH | Lee et al.) ──────────────────────
def panel(results_df, padj_col, reject_col, title, alpha=ALPHA):
    df = results_df.copy().sort_values("lfc")
    colors = ["red" if r else "steelblue" for r in df[reject_col]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["lfc"], y=df.index, orientation="h",
        error_x=dict(type="data", array=df["lfcSE"].abs().values, visible=True),
        marker_color=colors,
        text=[f"q={v:.3f}" for v in df[padj_col]],
        hovertemplate="%{y}<br>LFC=%{x:.3f}<br>%{text}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.update_layout(
        title=title,
        xaxis_title="Log2 Fold Change (smoker vs never-smoker)",
        yaxis_title="Family",
        height=max(400, 20 * len(df)),
        width=700,
        showlegend=False,
    )
    return fig

fig_bh  = panel(results, "padj_bh",  "reject_bh",
                f"AGP LinDA: BH adjustment (q < {ALPHA})")
fig_lee = panel(results, "q_lee",    "reject_lee",
                f"AGP LinDA: Lee et al. adjustment (q < {ALPHA})")

for fmt in ("png", "svg"):
    out_dir_fmt = os.path.join(REPO_ROOT, "plots", fmt)
    os.makedirs(out_dir_fmt, exist_ok=True)
    fig_bh.write_image( os.path.join(out_dir_fmt, f"da_family_agp_smoking_bh.{fmt}"))
    fig_lee.write_image(os.path.join(out_dir_fmt, f"da_family_agp_smoking_lee.{fmt}"))

# Combined two-panel figure (matches KORA da_family_smoking.png layout)
fig_combined = make_subplots(rows=1, cols=2,
                              subplot_titles=[f"(a) BH (q<{ALPHA})",
                                              f"(b) Lee et al. (q<{ALPHA})"])
for trace in fig_bh.data:
    fig_combined.add_trace(trace, row=1, col=1)
for trace in fig_lee.data:
    fig_combined.add_trace(trace, row=1, col=2)
fig_combined.update_layout(
    title="AGP: Differential abundance at family level (LinDA)",
    height=max(400, 20 * len(results)),
    width=1400,
    showlegend=False,
)
for fmt in ("png", "svg"):
    out_dir_fmt = os.path.join(REPO_ROOT, "plots", fmt)
    fig_combined.write_image(os.path.join(out_dir_fmt, f"da_family_agp_smoking.{fmt}"))

print(f"\n=== Done. Outputs in {OUT_DIR} and plots/ ===")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
