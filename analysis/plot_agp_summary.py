"""
Generate all AGP summary figures to match the KORA plots in Chapter 5:

  1. Pair-matching balance: Age + BMI distributions pre/post matching
  2. Richness boxplot: observed family richness by smoking status
  3. Volcano plots: LinDA BH and Lee et al. corrections
  4. Venn diagrams: stability-selected families, BH-significant DA,
                    Lee-significant DA (AGP vs KORA)

All figures saved as PNG (300 dpi) and SVG in plots/png/ and plots/svg/.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from matplotlib_venn import venn2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(REPO_ROOT, "source", "design_AG")
RES_DIR   = os.path.join(REPO_ROOT, "results", "agp")
PNG_DIR   = os.path.join(REPO_ROOT, "plots", "png")
SVG_DIR   = os.path.join(REPO_ROOT, "plots", "svg")
for d in (PNG_DIR, SVG_DIR): os.makedirs(d, exist_ok=True)

# ── Colour palette (matches chapter-5 convention) ────────────────────────────
C_SMOKER  = "#E84646"   # red
C_NEVER   = "#4C9BE8"   # blue
C_SIG     = "#E84646"
C_NONSIG  = "#AAAAAA"

def save(fig, name):
    fig.savefig(os.path.join(PNG_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(SVG_DIR, f"{name}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .svg")

# ── Load raw data ─────────────────────────────────────────────────────────────
full     = pd.read_csv(os.path.join(DATA_DIR, "agdata_smoke.csv"), index_col=0)
meta_sm  = pd.read_csv(os.path.join(DATA_DIR, "sample_data_smoker.csv"),   index_col=0)
meta_ns  = pd.read_csv(os.path.join(DATA_DIR, "sample_data_non_smoker.csv"), index_col=0)
otu_sm   = pd.read_csv(os.path.join(DATA_DIR, "otu_table_smoker.csv"),    index_col=0)
otu_ns   = pd.read_csv(os.path.join(DATA_DIR, "otu_table_non_smoker.csv"), index_col=0)
linda    = pd.read_csv(os.path.join(RES_DIR, "da", "linda_agp_adjpvalues.csv"), index_col=0)
classif  = pd.read_csv(os.path.join(RES_DIR, "classification", "selected_families_agp.csv"))

sm_ids   = meta_sm.index
ns_ids   = meta_ns.index

pre_sm = full[full["smoking_frequency"] == "Daily"]
pre_ns = full[full["smoking_frequency"] == "Never"]
post_sm = full.loc[full.index.intersection(sm_ids)]
post_ns = full.loc[full.index.intersection(ns_ids)]

def safe_float(s):
    try: return float(s)
    except: return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAIR-MATCHING BALANCE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 1. Pair-matching balance ===")

age_pre_sm  = pd.to_numeric(pre_sm["age_years"],     errors="coerce").dropna()
age_pre_ns  = pd.to_numeric(pre_ns["age_years"],     errors="coerce").dropna()
bmi_pre_sm  = pd.to_numeric(pre_sm["bmi_corrected"], errors="coerce").dropna()
bmi_pre_ns  = pd.to_numeric(pre_ns["bmi_corrected"], errors="coerce").dropna()

age_post_sm = pd.to_numeric(post_sm["age_years"],     errors="coerce").dropna()
age_post_ns = pd.to_numeric(post_ns["age_years"],     errors="coerce").dropna()
bmi_post_sm = pd.to_numeric(post_sm["bmi_corrected"], errors="coerce").dropna()
bmi_post_ns = pd.to_numeric(post_ns["bmi_corrected"], errors="coerce").dropna()

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle("AGP: Covariate balance before and after matching", fontsize=13)

def kde_plot(ax, a, b, xlabel, title, la="Smoker", lb="Never-smoker"):
    for vals, col, lab in [(a, C_SMOKER, la), (b, C_NEVER, lb)]:
        kde = gaussian_kde(vals, bw_method=0.3)
        x   = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(x, kde(x), color=col, lw=2, label=lab)
        ax.fill_between(x, kde(x), alpha=0.15, color=col)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)

kde_plot(axes[0,0], age_pre_sm,  age_pre_ns,  "Age (years)",
         f"Age — before matching\n(n={len(pre_sm)} vs n={len(pre_ns):,})")
kde_plot(axes[0,1], bmi_pre_sm,  bmi_pre_ns,  "BMI (kg/m²)",
         f"BMI — before matching\n(n={len(pre_sm)} vs n={len(pre_ns):,})")
kde_plot(axes[1,0], age_post_sm, age_post_ns, "Age (years)",
         f"Age — after matching\n(n={len(post_sm)} vs n={len(post_ns)})")
kde_plot(axes[1,1], bmi_post_sm, bmi_post_ns, "BMI (kg/m²)",
         f"BMI — after matching\n(n={len(post_sm)} vs n={len(post_ns)})")

plt.tight_layout()
save(fig, "agp_matching_balance")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RICHNESS BOXPLOT (observed family richness per sample)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. Richness boxplot ===")

rich_sm = (otu_sm > 0).sum(axis=1).values
rich_ns = (otu_ns > 0).sum(axis=1).values

# Betta result annotation
betta = pd.read_csv(os.path.join(RES_DIR, "alpha_diversity", "richness_betta_table.csv"), index_col=0)
pval  = betta.loc["W", "p_value"]
coef  = betta.loc["W", "estimate"]

fig, ax = plt.subplots(figsize=(5, 6))
data = [rich_sm, rich_ns]
bp   = ax.boxplot(data, labels=["Smoker", "Never-smoker"],
                  patch_artist=True, widths=0.5,
                  medianprops=dict(color="black", lw=2))
for patch, col in zip(bp["boxes"], [C_SMOKER, C_NEVER]):
    patch.set_facecolor(col); patch.set_alpha(0.7)

ax.set_ylabel("Observed family richness", fontsize=11)
ax.set_title("AGP: Observed richness by smoking status", fontsize=11)
ax.text(0.5, 0.97, f"Breakaway betta: W = {coef:.2f},  p = {pval:.3f}",
        ha="center", va="top", transform=ax.transAxes, fontsize=9,
        style="italic", color="dimgray")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save(fig, "agp_richness_boxplot")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VOLCANO PLOTS (BH and Lee et al.)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. Volcano plots ===")

def volcano(ax, df, pval_col, reject_col, title, alpha=0.1):
    x = df["lfc"]
    y = -np.log10(df[pval_col].clip(lower=1e-10))
    colors = [C_SIG if r else C_NONSIG for r in df[reject_col]]
    ax.scatter(x, y, c=colors, s=50, alpha=0.8, edgecolors="none")
    thr = -np.log10(alpha)
    ax.axhline(thr, color="black", lw=1, ls="--", alpha=0.5)
    ax.axvline(0,  color="black", lw=0.8, alpha=0.4)
    for _, row in df[df[reject_col]].iterrows():
        ax.annotate(row.name, xy=(row["lfc"], -np.log10(row[pval_col])),
                    fontsize=6.5, ha="left" if row["lfc"] > 0 else "right",
                    xytext=(3 if row["lfc"] > 0 else -3, 2),
                    textcoords="offset points")
    ax.set_xlabel("Log₂ fold change (smoker vs never-smoker)", fontsize=10)
    ax.set_ylabel("−log₁₀(p-value)", fontsize=10)
    ax.set_title(title, fontsize=10)
    sig_patch  = mpatches.Patch(color=C_SIG,  label=f"q < {alpha}")
    ns_patch   = mpatches.Patch(color=C_NONSIG, label=f"q ≥ {alpha}")
    ax.legend(handles=[sig_patch, ns_patch], fontsize=8)
    ax.spines[["top","right"]].set_visible(False)

# BH volcano
fig, ax = plt.subplots(figsize=(7, 5))
volcano(ax, linda, "pvalue", "reject_bh",
        f"AGP LinDA: BH correction (q < 0.1, n={linda['reject_bh'].sum()} significant)")
plt.tight_layout()
save(fig, "agp_volcano_bh")

# Lee et al. volcano
fig, ax = plt.subplots(figsize=(7, 5))
volcano(ax, linda, "p_perm", "reject_lee",
        f"AGP LinDA: Lee et al. correction (q < 0.1, n={linda['reject_lee'].sum()} significant)")
plt.tight_layout()
save(fig, "agp_volcano_lee")

# Combined two-panel
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("AGP: Differential abundance at family level (LinDA)", fontsize=12)
volcano(axes[0], linda, "pvalue",  "reject_bh",
        f"(a) BH correction  (n={linda['reject_bh'].sum()} sig.)")
volcano(axes[1], linda, "p_perm",  "reject_lee",
        f"(b) Lee et al. correction  (n={linda['reject_lee'].sum()} sig.)")
plt.tight_layout()
save(fig, "agp_volcano_combined")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VENN DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 4. Venn diagrams ===")

# ── KORA known results (from Chapter 5 thesis) ───────────────────────────────
kora_stabsel = {"Bifidobacteriaceae", "Christensenellaceae", "Erysipelotrichaceae"}
# From tab:family_bh_lee_main — families with BH q < 0.1
kora_bh  = {"f__Izemoplasmatales", "f__Clostridia_vadinBB60_group",
            "f__UCG-010", "f__Christensenellaceae", "f__Erysipelatoclostridiaceae",
            "f__Erysipelotrichaceae"}
# From tab:family_bh_lee_main — families with Lee et al. q < 0.1
kora_lee = {"f__Izemoplasmatales", "f__Clostridia_vadinBB60_group", "f__UCG-010"}

# ── AGP results ───────────────────────────────────────────────────────────────
agp_stabsel = set(classif.loc[classif["selected"], "family"])
agp_bh      = set(linda[linda["reject_bh"]].index)
agp_lee     = set(linda[linda["reject_lee"]].index)

# Normalise KORA names (remove f__ prefix for comparison)
def norm(s): return s.replace("f__","").strip()
kora_bh_n  = {norm(x) for x in kora_bh}
kora_lee_n = {norm(x) for x in kora_lee}

def make_venn(ax, set_a, set_b, lab_a, lab_b, title):
    v = venn2([set_a, set_b], set_labels=(lab_a, lab_b), ax=ax,
              set_colors=(C_SMOKER, C_NEVER), alpha=0.5)
    inter = set_a & set_b
    if v.get_label_by_id("11") and inter:
        v.get_label_by_id("11").set_text("\n".join(sorted(inter)))
        v.get_label_by_id("11").set_fontsize(7)
    ax.set_title(title, fontsize=9, pad=8)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("AGP vs KORA: cross-dataset overlap", fontsize=12)

make_venn(axes[0], agp_stabsel, kora_stabsel,
          f"AGP (n={len(agp_stabsel)})", f"KORA (n={len(kora_stabsel)})",
          "Stability-selected\nfamilies (P ≥ 0.65)")

make_venn(axes[1], agp_bh, kora_bh_n,
          f"AGP (n={len(agp_bh)})", f"KORA (n={len(kora_bh_n)})",
          "DA families\n(BH q < 0.1)")

make_venn(axes[2], agp_lee, kora_lee_n,
          f"AGP (n={len(agp_lee)})", f"KORA (n={len(kora_lee_n)})",
          "DA families\n(Lee et al. q < 0.1)")

plt.tight_layout()
save(fig, "agp_kora_venn")

# Individual Venns (for thesis individual figure inclusion)
for sets, name, title in [
    ((agp_stabsel, kora_stabsel), "venn_stabsel",
     "Stability-selected families (classo, P ≥ 0.65)"),
    ((agp_bh, kora_bh_n),        "venn_bh",
     "Differentially abundant families (LinDA, BH q < 0.1)"),
    ((agp_lee, kora_lee_n),       "venn_lee",
     "Differentially abundant families (LinDA, Lee et al. q < 0.1)"),
]:
    fig, ax = plt.subplots(figsize=(5, 4))
    make_venn(ax, sets[0], sets[1],
              f"AGP (n={len(sets[0])})", f"KORA (n={len(sets[1])})", title)
    plt.tight_layout()
    save(fig, f"agp_kora_{name}")

print("\n=== All plots done ===")
print("PNG:", PNG_DIR)
print("SVG:", SVG_DIR)
