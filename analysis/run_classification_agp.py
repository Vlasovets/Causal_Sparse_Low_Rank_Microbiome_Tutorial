"""
AGP predictive modeling: log-contrast classification (classo) with stability selection.
Replicates the KORA approach (notebooks/05_classification.ipynb) on the matched AGP
smoking dataset at family level.

Inputs (relative to repo root):
  source/design_AG/otu_table_smoker.csv        -- 234 x 40 counts
  source/design_AG/otu_table_non_smoker.csv    -- 234 x 40 counts
  source/design_AG/sample_data_smoker.csv      -- pair_nb, W
  source/design_AG/sample_data_non_smoker.csv
  source/design_AG/tax_table_smoker.csv        -- OTU -> Family

Outputs (results/agp/classification/):
  classification_agp_results.csv    -- MCR table (LAMfixed / CV / StabSel)
  selected_families_agp.csv         -- stability-selected families + probabilities
  smoking_classification_agp.png/svg -- stability probabilities + MCR figure
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split

from classo import classo_problem, clr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(REPO_ROOT, "source", "design_AG")
OUT_DIR   = os.path.join(REPO_ROOT, "results", "agp", "classification")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Hyperparameters (identical to KORA notebook) ──────────────────────────────
SEED             = 42
STABSEL_THRESH   = 0.65
NLAM             = 160
N_CV_SPLITS      = 5
PSEUDO_COUNT     = 1
TEST_SIZE        = 0.2

# ── Load data ─────────────────────────────────────────────────────────────────
otu_sm  = pd.read_csv(os.path.join(DATA_DIR, "otu_table_smoker.csv"),     index_col=0)
otu_ns  = pd.read_csv(os.path.join(DATA_DIR, "otu_table_non_smoker.csv"), index_col=0)
meta_sm = pd.read_csv(os.path.join(DATA_DIR, "sample_data_smoker.csv"),   index_col=0)
meta_ns = pd.read_csv(os.path.join(DATA_DIR, "sample_data_non_smoker.csv"), index_col=0)
tax     = pd.read_csv(os.path.join(DATA_DIR, "tax_table_smoker.csv"),     index_col=0)

# Family name map
family_map = {str(idx): (fam if pd.notna(fam) else f"f__unknown_{idx}")
              for idx, fam in tax["Family"].items()}

meta_sm["W"] = 1
meta_ns["W"] = 0
otu  = pd.concat([otu_sm, otu_ns], axis=0)
meta = pd.concat([meta_sm[["pair_nb","W"]], meta_ns[["pair_nb","W"]]], axis=0)
meta = meta.loc[otu.index]

# Rename OTU columns to family names
otu.columns = [family_map.get(str(c), str(c)) for c in otu.columns]

# Remove zero-variance columns
otu = otu.loc[:, otu.var() > 0]
family_names = otu.columns.tolist()
print(f"Samples: {otu.shape[0]}, Families: {otu.shape[1]}")

# ── Prepare features and labels ───────────────────────────────────────────────
# CLR transform: clr() expects (features x samples), returns (features x samples)
X_clr = clr(otu.values.T, PSEUDO_COUNT).T   # samples x families
y = np.where(meta["W"].values == 0, -1, 1)   # -1 = never-smoker, +1 = smoker

print(f"X_clr shape: {X_clr.shape}, y unique: {np.unique(y, return_counts=True)}")

# Train / test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_clr, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

# Log-contrast constraint: sum(beta) = 0
C = np.ones((1, X_train.shape[1]))

# ── Solve classo ──────────────────────────────────────────────────────────────
print("\n=== classo: stability selection + CV ===")
problem = classo_problem(X_train, y_train, C, label=family_names)
problem.formulation.classification = True
problem.formulation.intercept      = True
problem.formulation.concomitant    = False
problem.formulation.huber          = False

problem.model_selection.LAMfixed  = True
problem.model_selection.CV        = True
problem.model_selection.PATH      = True
problem.model_selection.StabSel   = True

problem.model_selection.CVparameters.seed    = SEED
problem.model_selection.CVparameters.Nsubset = N_CV_SPLITS
problem.model_selection.PATHparameters.Nlam  = NLAM
problem.model_selection.PATHparameters.numerical_method = "Path-Alg"
problem.model_selection.StabSelparameters.threshold = STABSEL_THRESH
problem.model_selection.StabSelparameters.method    = "first"
problem.model_selection.LAMfixedparameters.lam      = "theoretical"

problem.solve()
sol = problem.solution

# ── MCR ───────────────────────────────────────────────────────────────────────
def mcr(beta, X, y):
    y_hat = np.sign(X @ beta.reshape(-1, 1))
    y_hat[y_hat == 0] = 1
    return float(np.mean(y_hat.flatten() != y))

# Handle intercept prepended by classo
def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

X_te = add_intercept(X_test)  if sol.LAMfixed.beta.shape[0] == X_test.shape[1]  + 1 else X_test
X_tr = add_intercept(X_train) if sol.LAMfixed.beta.shape[0] == X_train.shape[1] + 1 else X_train

mcr_table = pd.DataFrame({
    "method":  ["LAMfixed", "CV",        "StabSel"],
    "MCR_out": [mcr(sol.LAMfixed.beta, X_te, y_test),
                mcr(sol.CV.beta,       X_te, y_test),
                mcr(sol.StabSel.refit, X_te, y_test)],
    "MCR_in":  [mcr(sol.LAMfixed.beta, X_tr, y_train),
                mcr(sol.CV.beta,       X_tr, y_train),
                mcr(sol.StabSel.refit, X_tr, y_train)],
})
print("\nMisclassification rates:")
print(mcr_table.to_string(index=False))
mcr_table.to_csv(os.path.join(OUT_DIR, "classification_agp_results.csv"), index=False)

# ── Stability-selected families ───────────────────────────────────────────────
n_feat = len(family_names)

# selected_param / distribution may include intercept as first entry
selected_mask = np.array(sol.StabSel.selected_param)
if len(selected_mask) == n_feat + 1:   # intercept prepended → drop it
    selected_mask = selected_mask[1:]

stab_probs = sol.StabSel.distribution
if stab_probs is not None:
    probs_arr = np.array(stab_probs)
    if probs_arr.ndim == 2:
        max_probs = probs_arr.max(axis=0)
    else:
        max_probs = probs_arr
    if len(max_probs) == n_feat + 1:
        max_probs = max_probs[1:]
    max_probs = max_probs[:n_feat]
else:
    max_probs = np.zeros(n_feat)

sel_df = pd.DataFrame({
    "family":    family_names,
    "stab_prob": max_probs,
    "selected":  selected_mask[:n_feat],
}).sort_values("stab_prob", ascending=False)

print(f"\nStability-selected families (threshold={STABSEL_THRESH}):")
print(sel_df[sel_df["selected"]][["family","stab_prob"]].to_string(index=False))
sel_df.to_csv(os.path.join(OUT_DIR, "selected_families_agp.csv"), index=False)

# ── Extract and save refit coefficients ──────────────────────────────────────
refit_beta = np.array(sol.StabSel.refit) if sol.StabSel.refit is not None else np.array([])

# Diagnostics: show full refit vector
print(f"\n--- refit_beta diagnostics ---")
print(f"len(refit_beta)={len(refit_beta)},  n_feat={n_feat}")
nz = np.where(np.abs(refit_beta) > 1e-8)[0]
print(f"Non-zero positions: {nz}, values: {refit_beta[nz]}")

# Selected features in their ORIGINAL column order (matches refit_beta ordering)
selected_indices       = np.where(selected_mask[:n_feat])[0]
sel_families_ordered   = [family_names[i] for i in selected_indices]
n_sel                  = len(sel_families_ordered)
print(f"selected_indices={selected_indices}, sel_families={sel_families_ordered}")

# refit vector layout: [intercept, coef_feature_orig_order_1, ..., coef_feature_orig_order_k]
# OR: [coef_feature_1, ..., coef_feature_40] (full vector, no intercept)
# Detect: if len == n_feat+1 it includes intercept; if len == n_feat it's all features
if len(refit_beta) == n_feat + 1:      # full feature set + intercept
    intercept_val = float(refit_beta[0])
    coef_vals     = refit_beta[1:][selected_indices]   # extract selected by index
elif len(refit_beta) == n_feat:        # full feature set, no intercept
    intercept_val = 0.0
    coef_vals     = refit_beta[selected_indices]
elif len(refit_beta) == n_sel + 1:     # selected only + intercept
    intercept_val = float(refit_beta[0])
    coef_vals     = refit_beta[1:]
elif len(refit_beta) == n_sel:         # selected only, no intercept
    intercept_val = 0.0
    coef_vals     = refit_beta
else:
    print(f"WARNING: unexpected refit_beta length {len(refit_beta)}")
    intercept_val = 0.0
    coef_vals     = refit_beta[:n_sel]

print(f"\nRefit coefficients (original feature order):")
for fam, c in zip(sel_families_ordered, coef_vals):
    print(f"  {fam}: {c:.4f}")
print(f"  intercept: {intercept_val:.4f}")
print(f"  sum(feature coefs) = {sum(coef_vals):.6f}  [should be ~0 for log-contrast]")

refit_df = pd.DataFrame({
    "term":  ["intercept"] + sel_families_ordered,
    "coef":  [intercept_val] + list(coef_vals),
})
refit_df.to_csv(os.path.join(OUT_DIR, "refit_coefficients_agp.csv"), index=False)

# ── Figure 5.6-style: vertical bars, matplotlib ───────────────────────────────
C_SEL   = "#F08080"   # salmon  — Selected True
C_NOSEL = "#5C4B8A"   # purple  — Selected False
C_COEF  = "#2C3E6B"   # navy    — refit coefficients
THRESH  = STABSEL_THRESH

# Sort panel (a) by stab_prob descending
df_a = sel_df.sort_values("stab_prob", ascending=False).reset_index(drop=True)

fig, (ax_a, ax_b) = plt.subplots(
    2, 1, figsize=(12, 9),
    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.55}
)

# ── Panel (a): selection probabilities ────────────────────────────────────────
bar_cols = [C_SEL if s else C_NOSEL for s in df_a["selected"]]
ax_a.bar(range(len(df_a)), df_a["stab_prob"], color=bar_cols, width=0.7, zorder=3)
ax_a.axhline(THRESH, color="red", lw=1.5, zorder=4)
ax_a.text(len(df_a) - 0.5, THRESH + 0.02, f"P={THRESH}",
          color="red", ha="right", va="bottom", fontsize=10)
ax_a.set_xticks(range(len(df_a)))
ax_a.set_xticklabels(df_a["family"], rotation=45, ha="right", fontsize=8.5)
ax_a.set_ylim(0, 1.05)
ax_a.set_ylabel("Selection probability", fontsize=11)
ax_a.set_title("Stability selection profile at the family level", fontsize=12, pad=8)
ax_a.yaxis.grid(True, color="lightgray", lw=0.7, zorder=0)
ax_a.set_axisbelow(True)
ax_a.spines[["top","right"]].set_visible(False)
ax_a.text(-0.02, 1.02, "a", transform=ax_a.transAxes,
          fontsize=14, fontweight="bold", va="top")
legend_handles = [mpatches.Patch(color=C_SEL, label="True"),
                  mpatches.Patch(color=C_NOSEL, label="False")]
ax_a.legend(handles=legend_handles, title="Selected:", title_fontsize=9,
            fontsize=9, loc="upper right", frameon=False)

# ── Panel (b): refit coefficients ─────────────────────────────────────────────
b_labels = refit_df["term"].tolist()
b_vals   = refit_df["coef"].tolist()
ax_b.bar(range(len(b_labels)), b_vals, color=C_COEF, width=0.55, zorder=3)
ax_b.axhline(0, color="black", lw=0.8, zorder=4)
ax_b.set_xticks(range(len(b_labels)))
ax_b.set_xticklabels(b_labels, rotation=45, ha="right", fontsize=9)
ax_b.set_ylabel("Coefficient βi", fontsize=11)
ax_b.set_title("Refitted coefficients after stability selection", fontsize=12, pad=8)
ax_b.yaxis.grid(True, color="lightgray", lw=0.7, zorder=0)
ax_b.set_axisbelow(True)
ax_b.spines[["top","right"]].set_visible(False)
ax_b.text(-0.02, 1.06, "b", transform=ax_b.transAxes,
          fontsize=14, fontweight="bold", va="top")

mcr_cv_val  = float(mcr_table[mcr_table["method"]=="CV"]["MCR_out"].iloc[0])
mcr_sel_val = float(mcr_table[mcr_table["method"]=="StabSel"]["MCR_out"].iloc[0])
fig.text(0.5, 0.01,
         f"Out-of-sample MCR — CV: {mcr_cv_val:.3f}  |  StabSel: {mcr_sel_val:.3f}",
         ha="center", fontsize=10, color="dimgray", style="italic")

for fmt in ("png", "svg"):
    out_dir_fmt = os.path.join(REPO_ROOT, "plots", fmt)
    os.makedirs(out_dir_fmt, exist_ok=True)
    fig.savefig(os.path.join(out_dir_fmt, f"smoking_classification_agp.{fmt}"),
                dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n=== Done. Outputs in {OUT_DIR} and plots/ ===")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
