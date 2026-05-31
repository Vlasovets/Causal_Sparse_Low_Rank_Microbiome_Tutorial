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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# ── Figure: stability probabilities + MCR (matching KORA smoking_classification.png) ──
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["(a) Stability selection probabilities",
                    "(b) Refitted coefficients (selected families)"],
    column_widths=[0.6, 0.4],
)

# Panel (a): horizontal bar of stability probabilities
df_plot = sel_df.sort_values("stab_prob")
colors  = ["red" if s else "steelblue" for s in df_plot["selected"]]
fig.add_trace(go.Bar(
    x=df_plot["stab_prob"], y=df_plot["family"], orientation="h",
    marker_color=colors,
    hovertemplate="%{y}<br>P=%{x:.3f}<extra></extra>",
), row=1, col=1)
fig.add_vline(x=STABSEL_THRESH, line_dash="dash", line_color="black",
              annotation_text=f"P={STABSEL_THRESH}", annotation_position="top right",
              row=1, col=1)

# Panel (b): refit coefficients of selected families
sel_only = sel_df[sel_df["selected"]].copy()
if len(sel_only) > 0:
    # refit beta aligns to selected features
    refit_beta = sol.StabSel.refit
    if refit_beta is not None and len(refit_beta) >= len(sel_only):
        n_extra = 1 if len(refit_beta) == len(sel_only) + 1 else 0
        coefs = refit_beta[n_extra: n_extra + len(sel_only)]
        sel_only = sel_only.copy()
        sel_only["coef"] = coefs
        sel_only = sel_only.sort_values("coef")
        fig.add_trace(go.Bar(
            x=sel_only["coef"], y=sel_only["family"], orientation="h",
            marker_color=["red" if c > 0 else "steelblue" for c in sel_only["coef"]],
            hovertemplate="%{y}<br>coef=%{x:.4f}<extra></extra>",
        ), row=1, col=2)
        fig.add_vline(x=0, line_width=1, line_color="black", row=1, col=2)
else:
    fig.add_annotation(text="No families selected", row=1, col=2,
                       xref="paper", yref="paper", x=0.75, y=0.5,
                       showarrow=False, font=dict(size=14))

# MCR annotation
mcr_cv_val  = float(mcr_table[mcr_table["method"]=="CV"]["MCR_out"])
mcr_sel_val = float(mcr_table[mcr_table["method"]=="StabSel"]["MCR_out"])
fig.add_annotation(
    text=(f"Out-of-sample MCR<br>"
          f"CV: {mcr_cv_val:.3f} | StabSel: {mcr_sel_val:.3f}"),
    xref="paper", yref="paper", x=0.5, y=-0.08,
    showarrow=False, font=dict(size=12),
)

fig.update_layout(
    title="AGP: Log-contrast classification for smoking status (classo)",
    height=max(500, 20 * len(sel_df)),
    width=1200,
    showlegend=False,
)
fig.update_xaxes(title_text="Selection probability", row=1, col=1)
fig.update_xaxes(title_text="Coefficient", row=1, col=2)

for fmt in ("png", "svg"):
    out_dir_fmt = os.path.join(REPO_ROOT, "plots", fmt)
    os.makedirs(out_dir_fmt, exist_ok=True)
    fig.write_image(os.path.join(out_dir_fmt, f"smoking_classification_agp.{fmt}"))

print(f"\n=== Done. Outputs in {OUT_DIR} and plots/ ===")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
