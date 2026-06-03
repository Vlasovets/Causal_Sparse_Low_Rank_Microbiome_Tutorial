"""
Regenerate 3 KORA plots as both PNG and SVG:
  1. alpha_div_smoking      — richness + Shannon boxplots (KORA family level)
  2. da_family_smoking      — LinDA DA forest plot (KORA family level, BH + Lee)
  3. smoking_classification — classo stability + coefficients (KORA family level)

Outputs saved to results/kora/svg_exports/ (and thesis figures dir via separate copy step).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from classo import classo_problem, clr

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.pandas2ri as pandas2ri

REPO_ROOT  = '/home/itg/oleg.vlasovets/slr_example/Causal_Sparse_Low_Rank_Microbiome_Tutorial'
sys.path.insert(0, REPO_ROOT)   # must be set before any utils import
KORA_ROOT  = '/home/itg/oleg.vlasovets/slr_example/KORA_Smoking_SLR'
SRC        = os.path.join(KORA_ROOT, 'source')
OUT_DIR    = os.path.join(REPO_ROOT, 'results', 'kora', 'svg_exports')
THESIS_FIG = '/home/itg/oleg.vlasovets/thesis/figures/causality'
os.makedirs(OUT_DIR, exist_ok=True)

def r_to_pandas(r_df):
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_df)

def save(fig, stem):
    for fmt in ('png', 'svg'):
        path = os.path.join(OUT_DIR, f'{stem}.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    # Also copy to thesis
    for fmt in ('png', 'svg'):
        thesis_path = os.path.join(THESIS_FIG, f'{stem}.{fmt}')
        fig.savefig(thesis_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {stem}.{{png,svg}}")

# ── Load KORA family-level OTU + metadata ─────────────────────────────────────
print("Loading KORA family data ...")
otu_sm  = pd.read_csv(os.path.join(SRC, 'otu_table_smoker.csv'),     index_col=0)
otu_ns  = pd.read_csv(os.path.join(SRC, 'otu_table_non_smoker.csv'), index_col=0)
meta_sm = pd.read_csv(os.path.join(SRC, 'sample_data_smoker.csv'),   index_col=0)
meta_ns = pd.read_csv(os.path.join(SRC, 'sample_data_non_smoker.csv'), index_col=0)
meta_sm['W'] = 1;  meta_ns['W'] = 0
otu  = pd.concat([otu_sm, otu_ns], axis=0)
meta = pd.concat([meta_sm[['pair_nb','W']], meta_ns[['pair_nb','W']]], axis=0)
meta = meta.loc[otu.index]
family_names = otu.columns.tolist()
print(f"  {otu.shape[0]} samples × {otu.shape[1]} families")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ALPHA-DIVERSITY BOXPLOT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 1. alpha_div_smoking ===")
from scipy.stats import mannwhitneyu

# Observed richness and Shannon from family-level OTU
richness = (otu > 0).sum(axis=1)
rel      = otu.div(otu.sum(axis=1), axis=0).replace(0, np.nan)
shannon  = -( rel * np.log(rel) ).sum(axis=1)

sm_idx = meta[meta['W'] == 1].index
ns_idx = meta[meta['W'] == 0].index

rich_sm = richness.loc[sm_idx];  rich_ns = richness.loc[ns_idx]
shan_sm = shannon.loc[sm_idx];   shan_ns = shannon.loc[ns_idx]

stat_r, p_r = mannwhitneyu(rich_sm, rich_ns, alternative='two-sided')
stat_s, p_s = mannwhitneyu(shan_sm, shan_ns, alternative='two-sided')
print(f"  Richness: p={p_r:.4f}  Shannon: p={p_s:.4f}")

C_SM = '#E84646'; C_NS = '#4C9BE8'
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
fig.suptitle("KORA FF4: Alpha-diversity by smoking status", fontsize=12)

for ax, (sm, ns), ylabel, p_val in [
    (axes[0], (rich_sm, rich_ns), 'Observed family richness', p_r),
    (axes[1], (shan_sm, shan_ns), 'Shannon diversity (H\')',  p_s),
]:
    bp = ax.boxplot([sm, ns], labels=['Smoker','Non-Smoker'],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', lw=2))
    for patch, col in zip(bp['boxes'], [C_SM, C_NS]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    sig = '*' if p_val < 0.05 else 'ns'
    ax.set_title(f'p = {p_val:.4f}  ({sig})', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
save(fig, 'alpha_div_smoking')

# ═══════════════════════════════════════════════════════════════════════════════
# 2. KORA FAMILY DA — LinDA forest plot
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. da_family_smoking ===")

linda = importr("LinDA")

def r_to_pd(r_df):
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_df)

otu_T = otu.T.astype(int)
w_obs = meta[['W']].rename(columns={'W':'w'})
with localconverter(ro.default_converter + pandas2ri.converter):
    r_otu = ro.conversion.py2rpy(otu_T)
    r_w   = ro.conversion.py2rpy(w_obs)
r_otu.rownames = ro.StrVector(family_names)

lo  = linda.linda(r_otu, r_w, formula='~w', alpha=0.05, prev_cut=0.0, lib_cut=1)
out = r_to_pd(lo.rx2('output').rx2('w'))

df = pd.DataFrame({
    'family':    out.index,
    'lfc':       out['log2FoldChange'].values,
    'lfcSE':     out['lfcSE'].values,
    'pvalue':    out['pvalue'].values,
    'padj_bh':   out['padj'].values,
    'reject_bh': out['reject'].values,
}).set_index('family').sort_values('lfc')

# ── Permutation-based Lee et al. correction (500 perms for speed) ────────────
from utils.helper import (calculate_unadjusted_p_values,
                           min_unadjusted_p_values, adjusted_p_values)
obs_stat = pd.Series(out['stat'].values, index=out.index)
pair_ids = meta['pair_nb'].unique()
perm_stats = np.empty((len(obs_stat), 0))
for _ in range(500):
    w_p = meta[['W','pair_nb']].copy()
    for pid in pair_ids:
        mask = w_p['pair_nb'] == pid
        if np.random.rand() > .5: w_p.loc[mask,'W'] = 1 - w_p.loc[mask,'W']
    w_p = w_p[['W']].rename(columns={'W':'w'})
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_wp = ro.conversion.py2rpy(w_p)
    lp = linda.linda(r_otu, r_wp, formula='~w', alpha=0.05, prev_cut=0.0, lib_cut=1)
    op = r_to_pd(lp.rx2('output').rx2('w'))
    perm_stats = np.hstack([perm_stats, op['stat'].reindex(obs_stat.index).values.reshape(-1,1)])

perm_df  = pd.DataFrame(perm_stats, index=obs_stat.index)
pval_lee = calculate_unadjusted_p_values(perm_df, obs_stat, test_type='two-sided')
min_p    = min_unadjusted_p_values(perm_df)
adj_p    = adjusted_p_values(min_p, pval_lee.reset_index(drop=True))
df['q_lee'] = pd.Series(adj_p, index=pval_lee.index).reindex(df.index).values
df['p_perm'] = pval_lee.reindex(df.index).values
df['reject_lee'] = df['q_lee'] < 0.1
df = df.sort_values('lfc')
print(f"  BH q<0.1: {df['reject_bh'].sum()}  Lee q<0.1: {df['reject_lee'].sum()}")

C_SIG_HIGH = "#E84646"
C_SIG_MID  = "#9B59B6"
C_NONSIG   = "#AAAAAA"

def kora_volcano(ax, df, pval_col, qval_col, title, alpha=0.1, alpha2=0.2):
    x = df["lfc"]
    y = -np.log10(df[pval_col].clip(lower=1e-10))
    q = df[qval_col]
    colors = [C_SIG_HIGH if qi < alpha else (C_SIG_MID if qi < alpha2 else C_NONSIG)
              for qi in q]
    ax.scatter(x, y, c=colors, s=50, alpha=0.8, edgecolors="none")
    ax.axhline(-np.log10(alpha),  color="black",   lw=1,   ls="--", alpha=0.5)
    ax.axhline(-np.log10(alpha2), color=C_SIG_MID, lw=0.8, ls=":",  alpha=0.6)
    ax.axvline(0, color="black", lw=0.8, alpha=0.4)
    for idx, row in df[q < alpha2].iterrows():
        ax.annotate(idx, xy=(row["lfc"], -np.log10(row[pval_col])),
                    fontsize=6.5, ha="left" if row["lfc"] > 0 else "right",
                    xytext=(3 if row["lfc"] > 0 else -3, 2),
                    textcoords="offset points")
    ax.set_xlabel("Log₂ fold change (smoker vs never-smoker)", fontsize=10)
    ax.set_ylabel("−log₁₀(p-value)", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(handles=[
        mpatches.Patch(color=C_SIG_HIGH, label=f"q < {alpha}"),
        mpatches.Patch(color=C_SIG_MID,  label=f"{alpha} ≤ q < {alpha2}"),
        mpatches.Patch(color=C_NONSIG,   label=f"q ≥ {alpha2}"),
    ], fontsize=8)
    ax.spines[["top","right"]].set_visible(False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("KORA FF4: Differential abundance at family level (LinDA)", fontsize=11)
kora_volcano(ax1, df, "pvalue",  "padj_bh",
             f"(a) BH correction  (n={df['reject_bh'].sum()} sig.)")
kora_volcano(ax2, df, "p_perm",  "q_lee",
             f"(b) Lee et al. correction  (n={df['reject_lee'].sum()} sig.)")
plt.tight_layout()
save(fig, 'da_family_smoking')

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SMOKING CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. smoking_classification ===")
# Load taxonomy for KORA families
tax_f = os.path.join(SRC, 'tax_table.csv')
tax   = pd.read_csv(tax_f, index_col=0) if os.path.exists(tax_f) else pd.DataFrame()

otu_filt = otu.loc[:, otu.var() > 0]
fam_names = otu_filt.columns.tolist()
X_clr = clr(otu_filt.values.T, 1).T
y = np.where(meta['W'].values == 0, -1, 1)
X_tr, X_te, y_tr, y_te = train_test_split(X_clr, y, test_size=0.2,
                                            random_state=42, stratify=y)
C = np.ones((1, X_tr.shape[1]))
prob = classo_problem(X_tr, y_tr, C, label=fam_names)
prob.formulation.classification = True
prob.formulation.intercept      = True
prob.model_selection.StabSel    = True
prob.model_selection.CV         = True
prob.model_selection.PATH       = False
prob.model_selection.LAMfixed   = False
prob.model_selection.CVparameters.seed = 42
prob.model_selection.StabSelparameters.threshold = 0.65
prob.solve()
sol = prob.solution

# Extract stability probabilities
n_feat = len(fam_names)
sel_mask = np.array(sol.StabSel.selected_param)
if len(sel_mask) == n_feat + 1: sel_mask = sel_mask[1:]
probs_raw = sol.StabSel.distribution
if probs_raw is not None:
    pa = np.array(probs_raw)
    max_p = pa.max(axis=0) if pa.ndim == 2 else pa
    if len(max_p) == n_feat + 1: max_p = max_p[1:]
    max_p = max_p[:n_feat]
else:
    max_p = np.zeros(n_feat)

sel_df = pd.DataFrame({'family': fam_names, 'stab_prob': max_p,
                        'selected': sel_mask[:n_feat]}).sort_values('stab_prob', ascending=False)

# Refit coefficients
refit_beta = np.array(sol.StabSel.refit) if sol.StabSel.refit is not None else np.array([])
sel_idx    = np.where(sel_mask[:n_feat])[0]
sel_fams   = [fam_names[i] for i in sel_idx]
n_sel      = len(sel_fams)
has_int    = (len(refit_beta) == n_feat + 1)
int_val    = float(refit_beta[0]) if has_int else 0.0
coef_vals  = refit_beta[1:][sel_idx] if has_int else refit_beta[sel_idx]
print(f"  Selected (P≥0.65): {sel_fams}  sum={sum(coef_vals):.6f}")

def add_intercept(X): return np.hstack([np.ones((X.shape[0],1)), X])

beta_cv = np.array(sol.CV.beta).reshape(-1) if sol.CV.beta is not None else np.zeros(X_te.shape[1])
X_te_cv = add_intercept(X_te) if len(beta_cv) == X_te.shape[1]+1 else X_te
MCR_CV  = float(np.mean(np.sign(X_te_cv @ beta_cv) != y_te))

X_te_sel = add_intercept(X_te) if has_int else X_te
MCR_sel  = float(np.mean(np.sign(X_te_sel @ refit_beta.reshape(-1)) != y_te)) \
           if has_int and n_sel > 0 else 0.5

C_SEL  = '#F08080'; C_NOSEL = '#5C4B8A'; C_COEF = '#2C3E6B'
fig, (ax_a, ax_b) = plt.subplots(2, 1,
    figsize=(12, 9), gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.55})

# Panel (a) — stability profile in original feature order
df_a = pd.DataFrame({'family': fam_names, 'stab_prob': max_p,
                      'selected': sel_mask[:n_feat]}).reset_index(drop=True)
bar_cols = [C_SEL if s else C_NOSEL for s in df_a['selected']]
ax_a.bar(range(len(df_a)), df_a['stab_prob'], color=bar_cols, width=0.7, zorder=3)
ax_a.axhline(0.65, color='red', lw=1.5, zorder=4)
ax_a.text(len(df_a)-0.5, 0.67, 'P=0.65', color='red', ha='right', fontsize=9)
ax_a.set_xticks(range(len(df_a)))
ax_a.set_xticklabels(df_a['family'], rotation=45, ha='right', fontsize=8)
ax_a.set_ylim(0, 1.05)
ax_a.set_ylabel('Selection probability', fontsize=10)
ax_a.set_title('Stability selection profile at the family level', fontsize=11)
ax_a.yaxis.grid(True, color='lightgray', lw=0.7, zorder=0)
ax_a.set_axisbelow(True)
ax_a.spines[['top','right']].set_visible(False)
ax_a.text(-0.02, 1.02, 'a', transform=ax_a.transAxes, fontsize=13, fontweight='bold')
ax_a.legend(handles=[mpatches.Patch(color=C_SEL, label='True'),
                      mpatches.Patch(color=C_NOSEL, label='False')],
            title='Selected:', title_fontsize=8, fontsize=8, loc='upper right', frameon=False)

# Panel (b) — refit coefficients
b_labels = ['intercept'] + sel_fams
b_vals   = [int_val] + list(coef_vals)
ax_b.bar(range(len(b_labels)), b_vals, color=C_COEF, width=0.55, zorder=3)
ax_b.axhline(0, color='black', lw=0.8)
ax_b.set_xticks(range(len(b_labels)))
ax_b.set_xticklabels(b_labels, rotation=45, ha='right', fontsize=9)
ax_b.set_ylabel('Coefficient βi', fontsize=10)
ax_b.set_title('Refitted coefficients after stability selection', fontsize=11)
ax_b.yaxis.grid(True, color='lightgray', lw=0.7, zorder=0)
ax_b.set_axisbelow(True)
ax_b.spines[['top','right']].set_visible(False)
ax_b.text(-0.02, 1.06, 'b', transform=ax_b.transAxes, fontsize=13, fontweight='bold')

fig.text(0.5, 0.01,
         f'Out-of-sample MCR — StabSel: {MCR_sel:.3f}',
         ha='center', fontsize=10, color='dimgray', style='italic')
save(fig, 'smoking_classification')

print("\nAll 3 KORA plots saved.")
