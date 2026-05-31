"""
KORA FF4 genus-level differential abundance using LinDA.
Enables direct method comparison with Alice's DACOMP genus-level results
(Sommer et al. 2022, Supplementary).

Inputs (KORA_Smoking_SLR/source/):
  otu_genus_smoker.csv        -- 236 x 211 genus counts (samples x genera)
  otu_genus_non_smoker.csv    -- 236 x 211
  sample_data_smoker.csv      -- metadata incl. pair_nb (last col)
  sample_data_non_smoker.csv

Outputs (results/kora/da_genus/):
  linda_kora_genus_results.csv    -- LFC, SE, stat, BH q per genus
  linda_kora_genus_adjpvalues.csv -- + Lee et al. q
  linda_kora_genus_permstats.csv  -- permutation statistics
  plots/png/da_genus_kora_smoking.png / .svg
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.pandas2ri as pandas2ri

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from utils.helper import (calculate_unadjusted_p_values,
                           min_unadjusted_p_values,
                           adjusted_p_values)

def r_to_pandas(r_df):
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_df)

# ── Config ────────────────────────────────────────────────────────────────────
KORA_DIR = os.path.join(os.path.dirname(REPO_ROOT), "KORA_Smoking_SLR", "source")
OUT_DIR  = os.path.join(REPO_ROOT, "results", "kora", "da_genus")
PNG_DIR  = os.path.join(REPO_ROOT, "plots", "png")
SVG_DIR  = os.path.join(REPO_ROOT, "plots", "svg")
for d in (OUT_DIR, PNG_DIR, SVG_DIR):
    os.makedirs(d, exist_ok=True)

N_PERM = 1000
ALPHA  = 0.1

linda = importr("LinDA")

# ── Load data ─────────────────────────────────────────────────────────────────
otu_sm  = pd.read_csv(os.path.join(KORA_DIR, "otu_genus_smoker.csv"),     index_col=0)
otu_ns  = pd.read_csv(os.path.join(KORA_DIR, "otu_genus_non_smoker.csv"), index_col=0)
meta_sm = pd.read_csv(os.path.join(KORA_DIR, "sample_data_smoker.csv"),   index_col=0)
meta_ns = pd.read_csv(os.path.join(KORA_DIR, "sample_data_non_smoker.csv"), index_col=0)

# Smoking indicator
meta_sm["W"] = 1
meta_ns["W"] = 0

# Intersect genera present in both groups (replicates KORA report preprocessing)
common_genera = otu_sm.columns.intersection(otu_ns.columns)
otu_sm = otu_sm[common_genera]
otu_ns = otu_ns[common_genera]

# Further filter: keep genera present in >0 samples in each group
prev_sm = (otu_sm > 0).sum(axis=0)
prev_ns = (otu_ns > 0).sum(axis=0)
keep    = common_genera[(prev_sm > 0) & (prev_ns > 0)]
otu_sm  = otu_sm[keep]
otu_ns  = otu_ns[keep]

otu  = pd.concat([otu_sm, otu_ns], axis=0)
meta = pd.concat([meta_sm[["pair_nb", "W"]], meta_ns[["pair_nb", "W"]]], axis=0)
meta = meta.loc[otu.index]

print(f"OTU table: {otu.shape}  (samples x genera)")
print(f"Genera after intersection + prevalence filter: {otu.shape[1]}")

# Transpose: genera x samples for LinDA
otu_T = otu.T.astype(int)
w_obs = meta[["W"]].rename(columns={"W": "w"})

# ── LinDA: observed assignment ────────────────────────────────────────────────
print("\n=== LinDA: observed assignment ===")
with localconverter(ro.default_converter + pandas2ri.converter):
    r_otu = ro.conversion.py2rpy(otu_T)
    r_w   = ro.conversion.py2rpy(w_obs)
r_otu.rownames = ro.StrVector(otu_T.index.tolist())

lo  = linda.linda(r_otu, r_w, formula="~w", alpha=ALPHA, prev_cut=0.0, lib_cut=1)
out = r_to_pandas(lo.rx2("output").rx2("w"))

results = pd.DataFrame({
    "genus":     out.index,
    "lfc":       out["log2FoldChange"].values,
    "lfcSE":     out["lfcSE"].values,
    "stat":      out["stat"].values,
    "pvalue":    out["pvalue"].values,
    "padj_bh":   out["padj"].values,
    "reject_bh": out["reject"].values,
}).set_index("genus")

results.to_csv(os.path.join(OUT_DIR, "linda_kora_genus_results.csv"))
n_bh = results["reject_bh"].sum()
print(f"Significant (BH q<{ALPHA}): {n_bh} of {len(results)} genera")
if n_bh > 0:
    print(results[results["reject_bh"]].sort_values("padj_bh")[["lfc","pvalue","padj_bh"]])

# ── Lee et al. permutation correction ────────────────────────────────────────
print(f"\n=== Lee et al. ({N_PERM} within-pair permutations) ===")
pair_ids   = meta["pair_nb"].unique()
perm_stats = np.empty((len(results), 0))

for i in range(N_PERM):
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{N_PERM}")
    w_perm = meta[["W", "pair_nb"]].copy()
    for pid in pair_ids:
        mask = w_perm["pair_nb"] == pid
        if np.random.rand() > 0.5:
            w_perm.loc[mask, "W"] = 1 - w_perm.loc[mask, "W"]
    w_perm = w_perm[["W"]].rename(columns={"W": "w"})
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_w_perm = ro.conversion.py2rpy(w_perm)
    lo_p  = linda.linda(r_otu, r_w_perm, formula="~w", alpha=ALPHA, prev_cut=0.0, lib_cut=1)
    out_p = r_to_pandas(lo_p.rx2("output").rx2("w"))
    perm_stats = np.hstack([perm_stats,
                             out_p["stat"].reindex(results.index).values.reshape(-1, 1)])

perm_df   = pd.DataFrame(perm_stats, index=results.index)
obs_stat  = results["stat"]
pval_lee  = calculate_unadjusted_p_values(perm_df, obs_stat, test_type="two-sided")
min_pvals = min_unadjusted_p_values(perm_df)
adj_pvals = adjusted_p_values(min_pvals, pval_lee.reset_index(drop=True))

adj_df = pval_lee.copy()
adj_df.columns = ["p_perm"]
adj_df["q_lee"]      = adj_pvals
adj_df["reject_lee"] = adj_df["q_lee"] < ALPHA
results = results.join(adj_df)
results.to_csv(os.path.join(OUT_DIR, "linda_kora_genus_adjpvalues.csv"))
perm_df.to_csv(os.path.join(OUT_DIR, "linda_kora_genus_permstats.csv"))

n_lee = results["reject_lee"].sum()
print(f"Significant (BH q<{ALPHA}):       {n_bh} genera")
print(f"Significant (Lee et al. q<{ALPHA}): {n_lee} genera")
if n_lee > 0:
    print(results[results["reject_lee"]].sort_values("q_lee")[["lfc","padj_bh","q_lee"]])

# ── Volcano plot ──────────────────────────────────────────────────────────────
C_SIG   = "#E84646"
C_NOSIG = "#AAAAAA"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("KORA FF4: LinDA differential abundance at the genus level", fontsize=12)

for ax, padj_col, reject_col, title in [
    (axes[0], "padj_bh",  "reject_bh",  f"(a) BH  (n={n_bh} sig.)"),
    (axes[1], "p_perm",   "reject_lee", f"(b) Lee et al.  (n={n_lee} sig.)"),
]:
    x = results["lfc"]
    y = -np.log10(results[padj_col].clip(lower=1e-10))
    colors = [C_SIG if r else C_NOSIG for r in results[reject_col]]
    ax.scatter(x, y, c=colors, s=25, alpha=0.8, edgecolors="none")
    ax.axhline(-np.log10(ALPHA), color="black", lw=1, ls="--", alpha=0.5)
    ax.axvline(0, color="black", lw=0.8, alpha=0.4)
    for genus, row in results[results[reject_col]].iterrows():
        ax.annotate(genus, xy=(row["lfc"], -np.log10(row[padj_col])),
                    fontsize=6, ha="left" if row["lfc"] > 0 else "right",
                    xytext=(3 if row["lfc"] > 0 else -3, 2),
                    textcoords="offset points")
    ax.set_xlabel("Log₂ fold change (smoker vs never-smoker)", fontsize=10)
    ax.set_ylabel("−log₁₀(q-value)", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
for fmt, d in [("png", PNG_DIR), ("svg", SVG_DIR)]:
    fig.savefig(os.path.join(d, f"da_genus_kora_smoking.{fmt}"),
                dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n=== Done. Outputs in {OUT_DIR} ===")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
