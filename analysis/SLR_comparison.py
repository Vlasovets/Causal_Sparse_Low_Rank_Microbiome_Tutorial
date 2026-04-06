#!/usr/bin/env python3
"""
Phase 2: SLR Diagnostic Comparison — Python (GGLasso/ADMM) vs. SPIEC-EASI
See analysis/SLR_diagnostic_plan.md for full context.

Each section isolates one root cause from the revised ranking, quantifies the
divergence it causes, applies the minimal fix to the Python side (without touching
any source file), and reports the Frobenius norm before and after.

Usage:
    # from project root:
    python analysis/SLR_comparison.py            # run all sections
    python analysis/SLR_comparison.py --section 1  # run a single section

Sections:
    1  Solver-level baseline         (bypass StARS; lower-bound divergence)
    2  Fix StARS beta                CAUSE #1: beta=0.05 → beta=0
    3  Fix instability normalization CAUSE #2: 2x-inflated formula → correct
    4  Fix CLR pseudocount           CAUSE #3: multiplicative → uniform +1
    5  Fix lambda grid spacing       CAUSE #5: log-spaced → linear-spaced
    6  Fix covariance bias           CAUSE #6: 1/n → 1/(n-1)
    7  Cumulative fix validation     all fixes simultaneously
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.special import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.solver import ADMM_single

try:
    from gglasso.helper.utils import sparsity as _gglasso_sparsity
    def sparsity(M):
        return _gglasso_sparsity(M)
except ImportError:
    def sparsity(M):
        p = M.shape[0]
        off = ~np.eye(p, dtype=bool)
        return float(np.sum(np.abs(M[off]) < 1e-8) / off.sum())

# ── output directories ────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_ROOT, "data", "AGP")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "analysis", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── fixed hyperparameters ─────────────────────────────────────────────────────
RANK   = 10      # matches SE r=10
MU1    = 1       # nuclear-norm weight; ignored when r is given
TOL    = 1e-5
RTOL   = 1e-5
N_SUB  = 10      # Python original subsample count
CASES  = ["smoker", "non_smoker"]

# ── SE lambda grids  (log-spaced; taken verbatim from SPIEC-EASI R printout) ─
LAMBDA_GRID = {
    "non_smoker": [
        3.02721826, 2.37563971, 1.86430695, 1.46303347, 1.14813010,
        0.90100654, 0.70707387, 0.55488327, 0.43545018, 0.34172387,
        0.26817121, 0.21045003, 0.16515276, 0.12960528, 0.10170903,
        0.07981718, 0.06263733, 0.04915527, 0.03857509, 0.03027218,
    ],
    "smoker": [
        4.66364059, 3.65983845, 2.87209471, 2.25390496, 1.76877439,
        1.38806334, 1.08929654, 0.85483632, 0.67084133, 0.52644942,
        0.41313643, 0.32421293, 0.25442933, 0.19966596, 0.15668985,
        0.12296392, 0.09649716, 0.07572711, 0.05942761, 0.04663641,
    ],
}


###############################################################################
# ── CLR helpers (defined locally to avoid utils/helper.py's rpy2 import) ────
###############################################################################

def _geomean_per_sample(X: pd.DataFrame) -> pd.Series:
    """Geometric mean per column (sample) of a p×n DataFrame."""
    return X.apply(lambda col: np.exp(np.mean(np.log(col.values))))


def clr_python(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Python-style CLR (mirrors utils/helper.py transform_features):
      1. Zero-imputation: replace zeros with pseudo_count=1, then rescale each
         column sum to its original value (multiplicative replacement).
      2. Normalize to relative abundances.
      3. log(X / geomean_per_sample(X)).
    Input:  p×n DataFrame of raw integer counts.
    Output: p×n DataFrame of CLR values.
    """
    X = counts.copy().astype(float)
    # step 1: multiplicative zero replacement
    for col in X.columns:
        orig_sum = X[col].sum()
        mask = X[col] == 0
        if mask.any():
            X.loc[mask, col] = 1.0
            new_sum = X[col].sum()
            if new_sum > 0:
                X[col] *= orig_sum / new_sum
    # step 2: normalize to simplex
    X = X / X.sum(axis=0)
    # step 3: CLR
    g = _geomean_per_sample(X)
    return np.log(X / g)


def clr_se(counts: pd.DataFrame) -> pd.DataFrame:
    """
    SPIEC-EASI-style CLR (R: t(clr(data + 1, 1))):
      Add a uniform pseudocount of +1 to EVERY entry, then CLR per sample.
    Input:  p×n DataFrame of raw integer counts.
    Output: p×n DataFrame of CLR values.
    """
    X = counts.copy().astype(float) + 1.0
    g = _geomean_per_sample(X)
    return np.log(X / g)


###############################################################################
# ── covariance & lambda helpers ───────────────────────────────────────────────
###############################################################################

def empirical_cov(clr_df: pd.DataFrame, bias: bool = True) -> np.ndarray:
    """Empirical covariance of a p×n DataFrame. bias=True → 1/n, False → 1/(n-1)."""
    return np.cov(clr_df.values, bias=bias)


def get_max_cov(S: np.ndarray) -> float:
    """pulsar::getMaxCov — max absolute off-diagonal entry of a symmetric matrix."""
    p = S.shape[0]
    mask = ~np.eye(p, dtype=bool)
    return float(np.max(np.abs(S[mask])))


def linear_lambda_grid(S: np.ndarray, n_lambda: int = 20,
                       min_ratio: float = 1e-2) -> list:
    """
    Replicates getLamPath(max, max*ratio, n, log=FALSE) from pulsar R package:
    linearly-spaced grid from lambda_max down to lambda_max * min_ratio.
    lambda_max = getMaxCov(S).
    """
    lmax = get_max_cov(S)
    return list(np.linspace(lmax, lmax * min_ratio, n_lambda))


###############################################################################
# ── StARS instability formulas ────────────────────────────────────────────────
###############################################################################

def _instability_inline(estimates: list, p: int) -> float:
    """
    Inline formula from AGP_SLR.py (factor-of-2 inflated):
      2 * sum(2 * mu_ij * (1 - mu_ij)) / (p * (p-1))
    Range: [0, 2] for a fully unstable edge set.
    Equivalent to 2× the correct formula for symmetric matrices with zero diagonal.
    """
    mu = np.mean(estimates, axis=0)
    xi = 2.0 * mu * (1.0 - mu)
    return float(2.0 * np.sum(xi) / (p * (p - 1)))


def _instability_correct(estimates: list, p: int) -> float:
    """
    Correct StARS formula (matches stability_selection.py):
      sum(2 * mu_ij * (1 - mu_ij)) / C(p,2) / 2
    Range: [0, 1] for a fully unstable edge set.
    """
    mu = np.mean(estimates, axis=0)
    xi = 2.0 * mu * (1.0 - mu)
    return float(np.sum(xi) / comb(p, 2) / 2.0)


###############################################################################
# ── StARS runner ─────────────────────────────────────────────────────────────
###############################################################################

def run_stars(clr_values: np.ndarray,
              lambda_grid: list,
              n_sub: int = N_SUB,
              beta: float = 0.05,
              instability_fn=None,
              seed: int = None) -> dict:
    """
    Run StARS model selection over lambda_grid.

    Subsamples are drawn once and reused across all lambda values (matches the
    original notebook pattern).

    Returns
    -------
    dict with keys:
        D_b       : np.ndarray of instability values, one per lambda
        opt_idx   : int, selected lambda index (largest where D_b <= beta)
        lambda_star : float
    """
    if seed is not None:
        np.random.seed(seed)
    if instability_fn is None:
        instability_fn = _instability_inline

    p, n = clr_values.shape
    # subsample size — matches SPIEC-EASI / original notebook heuristic
    sample_ratio = 0.8 if n <= 144 else 10.0 * np.sqrt(n) / n
    n_sub_actual = int(np.floor(n * sample_ratio))

    # draw subsamples once for all lambda values
    subsamples = []
    for _ in range(n_sub):
        idx = np.random.choice(n, size=n_sub_actual, replace=False)
        subsamples.append(clr_values[:, idx])

    D_b = []
    for lam in lambda_grid:
        estimates = []
        for sub in subsamples:
            S_sub = np.cov(sub, bias=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol, _ = ADMM_single(
                    S=S_sub, lambda1=lam, mu1=MU1, r=RANK,
                    Omega_0=np.eye(p), verbose=False,
                    latent=True, tol=TOL, rtol=RTOL,
                )
            G = sol["Theta"].astype(bool).astype(int)
            estimates.append(G)
        D_b.append(instability_fn(estimates, p))

    D_b = np.array(D_b)
    indices = np.where(D_b <= beta)[0]
    opt_idx = int(np.max(indices)) if len(indices) > 0 else 0

    return {"D_b": D_b, "opt_idx": opt_idx, "lambda_star": lambda_grid[opt_idx]}


###############################################################################
# ── metrics ──────────────────────────────────────────────────────────────────
###############################################################################

def frobenius(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, "fro"))


def jaccard_edges(Theta_a: np.ndarray, Theta_b: np.ndarray,
                  tol: float = 1e-8) -> float:
    """Jaccard similarity of off-diagonal nonzero patterns."""
    p = Theta_a.shape[0]
    mask = ~np.eye(p, dtype=bool)
    a = np.abs(Theta_a[mask]) > tol
    b = np.abs(Theta_b[mask]) > tol
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    return inter / union if union > 0 else 0.0


###############################################################################
# ── data loaders ─────────────────────────────────────────────────────────────
###############################################################################

def load_se_solution(case: str) -> dict:
    """Load SPIEC-EASI Theta and L matrices from saved CSV outputs."""
    Theta = pd.read_csv(
        os.path.join(DATA_DIR, f"theta_{case}.csv"), index_col=0
    ).values.astype(float)
    L = pd.read_csv(
        os.path.join(DATA_DIR, f"low_rank_{case}.csv"), index_col=0
    ).values.astype(float)
    return {"Theta": Theta, "L": L, "Omega": Theta - L}


def load_counts(case: str) -> pd.DataFrame:
    """Load OTU table as a p×n DataFrame (taxa × samples)."""
    df = pd.read_csv(
        os.path.join(DATA_DIR, f"otu_table_{case}.csv"), index_col=0
    )
    return df.T   # original CSV is N×p; transpose to p×n


def refit(S: np.ndarray, lam: float) -> dict:
    """Run ADMM at a single (S, lambda) and return the solution dict."""
    p = S.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol, _ = ADMM_single(
            S=S, lambda1=lam, mu1=MU1, r=RANK,
            Omega_0=np.eye(p), verbose=False,
            latent=True, tol=TOL, rtol=RTOL,
        )
    return sol


###############################################################################
# ── figure helpers ────────────────────────────────────────────────────────────
###############################################################################

def _heatmap(ax, data: np.ndarray, title: str, vmax=None):
    if vmax is None:
        vmax = max(np.max(np.abs(data)), 1e-12)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _save(fig, name: str):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


# accumulate rows for the final summary table
_SUMMARY: list = []


def _record(cause: str, fix: str, case: str,
            frob_before: float, frob_after: float):
    _SUMMARY.append({
        "cause": cause, "fix": fix, "case": case,
        "frob_before": round(frob_before, 4),
        "frob_after":  round(frob_after,  4),
        "delta": round(frob_before - frob_after, 4),
    })


###############################################################################
# SECTION 1 — Solver-level baseline
# ─────────────────────────────────────────────────────────────────────────────
# Bypasses StARS entirely. Runs the Python ADMM at every lambda index using
# Python's covariance (np.cov, bias=True, multiplicative CLR) and compares to
# the pre-computed SE solution. Identifies which lambda index minimises
# ||Theta_py - Theta_se||_F and treats that as the solver lower-bound:
# the irreducible divergence remaining even when both solvers receive the same
# inputs at the same lambda.
###############################################################################

def section1_baseline():
    print("\n" + "=" * 70)
    print("SECTION 1 — Solver-level baseline  (lower-bound divergence target)")
    print("=" * 70)

    n_cases = len(CASES)
    fig, axes = plt.subplots(n_cases, 5, figsize=(24, 5 * n_cases))
    axes = np.atleast_2d(axes)

    for ri, case in enumerate(CASES):
        print(f"\n  [{case}]")
        se  = load_se_solution(case)
        raw = load_counts(case)
        clr = clr_python(raw)
        S   = empirical_cov(clr, bias=True)
        p   = S.shape[0]
        grid = LAMBDA_GRID[case]

        frob_theta_path, frob_omega_path = [], []
        best_idx, best_frob = 0, np.inf
        best_sol = None

        for idx, lam in enumerate(grid):
            sol = refit(S, lam)
            ft  = frobenius(sol["Theta"], se["Theta"])
            fo  = frobenius(sol["Omega"], se["Omega"])
            frob_theta_path.append(ft)
            frob_omega_path.append(fo)
            if ft < best_frob:
                best_frob, best_idx, best_sol = ft, idx, sol

        print(f"  SE-closest λ: index={best_idx}  λ={grid[best_idx]:.5f}")
        print(f"  ||Θ_py−Θ_se||_F (min)  = {best_frob:.4f}  ← solver lower bound")
        print(f"  ||Ω_py−Ω_se||_F at idx = {frob_omega_path[best_idx]:.4f}")
        print(f"  Jaccard edges          = {jaccard_edges(best_sol['Theta'], se['Theta']):.4f}")
        print(f"  sparsity py={sparsity(best_sol['Theta']):.4f}  se={sparsity(se['Theta']):.4f}")

        _record("solver baseline", "none (lower bound)", case, best_frob, best_frob)

        eig_py = np.sort(np.linalg.eigvalsh(best_sol["L"]))[::-1]
        eig_se = np.sort(np.linalg.eigvalsh(se["L"]))[::-1]

        ax = axes[ri]
        # [0] Frobenius vs lambda index
        ax[0].plot(frob_theta_path, "o-", label="||Θ_py−Θ_se||_F")
        ax[0].plot(frob_omega_path, "s--", label="||Ω_py−Ω_se||_F")
        ax[0].axvline(best_idx, color="r", lw=1.5, linestyle=":",
                      label=f"best idx={best_idx}")
        ax[0].set_xlabel("λ index"); ax[0].set_ylabel("Frobenius norm")
        ax[0].set_title(f"{case}: Frob vs λ index"); ax[0].legend(fontsize=7)

        # [1][2][3] diff heatmaps
        _heatmap(ax[1], best_sol["Theta"] - se["Theta"],
                 f"{case}: Θ_py−Θ_se  F={best_frob:.3f}")
        _heatmap(ax[2], best_sol["L"] - se["L"],
                 f"{case}: L_py−L_se")
        _heatmap(ax[3], best_sol["Omega"] - se["Omega"],
                 f"{case}: Ω_py−Ω_se  F={frob_omega_path[best_idx]:.3f}")

        # [4] eigenvalue spectra
        ax[4].plot(eig_py[:RANK + 2], "o-", label="Python")
        ax[4].plot(eig_se[:RANK + 2], "s--", label="SE")
        ax[4].set_xlabel("rank"); ax[4].set_ylabel("eigenvalue")
        ax[4].set_title(f"{case}: L eigenvalues")
        ax[4].legend()

    fig.suptitle("Section 1 — Solver baseline: Python ADMM vs SE at closest λ",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "section1_baseline.png")


###############################################################################
# SECTION 2 — Fix StARS β    [CAUSE #1]
# ─────────────────────────────────────────────────────────────────────────────
# Python uses beta=0.05; SPIEC-EASI uses beta=0.
# Both runs use the Python CLR (multiplicative), log-spaced grid, and the
# inline (2×-inflated) normalization formula to isolate this one variable.
###############################################################################

def section2_stars_beta(seed: int = 42):
    print("\n" + "=" * 70)
    print("SECTION 2 — Fix StARS β  [CAUSE #1: Python beta=0.05 → SE beta=0]")
    print("=" * 70)

    n_cases = len(CASES)
    fig, axes = plt.subplots(n_cases, 3, figsize=(17, 5 * n_cases))
    axes = np.atleast_2d(axes)

    for ri, case in enumerate(CASES):
        print(f"\n  [{case}]")
        se   = load_se_solution(case)
        raw  = load_counts(case)
        clr  = clr_python(raw)
        S    = empirical_cov(clr, bias=True)
        p    = S.shape[0]
        grid = LAMBDA_GRID[case]

        res_005 = run_stars(clr.values, grid, n_sub=N_SUB, beta=0.05,
                            instability_fn=_instability_inline, seed=seed)
        res_0   = run_stars(clr.values, grid, n_sub=N_SUB, beta=0.0,
                            instability_fn=_instability_inline, seed=seed)

        sol_005 = refit(S, res_005["lambda_star"])
        sol_0   = refit(S, res_0["lambda_star"])

        frob_before = frobenius(sol_005["Theta"], se["Theta"])
        frob_after  = frobenius(sol_0["Theta"],   se["Theta"])

        print(f"  beta=0.05 → idx={res_005['opt_idx']}  λ={res_005['lambda_star']:.5f}"
              f"  ||Θ−Θ_se||_F={frob_before:.4f}")
        print(f"  beta=0    → idx={res_0['opt_idx']}    λ={res_0['lambda_star']:.5f}"
              f"  ||Θ−Θ_se||_F={frob_after:.4f}")
        _record("StARS β (0.05 vs 0)", "beta=0", case, frob_before, frob_after)

        ax = axes[ri]
        x = list(range(len(grid)))
        ax[0].plot(x, res_005["D_b"], "o-", label="β=0.05 run")
        ax[0].plot(x, res_0["D_b"],   "s--", label="β=0 run", alpha=0.8)
        ax[0].axhline(0.05, color="tab:blue", lw=1, linestyle="--", alpha=0.5)
        ax[0].axhline(0.0,  color="tab:red",  lw=1, linestyle="--", alpha=0.5)
        ax[0].axvline(res_005["opt_idx"], color="tab:blue", lw=1.5,
                      linestyle=":", label=f"λ*(β=0.05)=idx {res_005['opt_idx']}")
        ax[0].axvline(res_0["opt_idx"],   color="tab:red",  lw=1.5,
                      linestyle=":", label=f"λ*(β=0)  =idx {res_0['opt_idx']}")
        ax[0].set_xlabel("λ index"); ax[0].set_ylabel("D_b (inline formula)")
        ax[0].set_title(f"{case}: D_b curves"); ax[0].legend(fontsize=7)

        _heatmap(ax[1], sol_005["Theta"] - se["Theta"],
                 f"{case}: Θ(β=0.05)−Θ_se\nFrob={frob_before:.3f}")
        _heatmap(ax[2], sol_0["Theta"] - se["Theta"],
                 f"{case}: Θ(β=0)−Θ_se\nFrob={frob_after:.3f}")

    fig.suptitle("Section 2 — Effect of StARS β threshold", fontsize=11)
    fig.tight_layout()
    _save(fig, "section2_stars_beta.png")


###############################################################################
# SECTION 3 — Fix instability normalization    [CAUSE #2]
# ─────────────────────────────────────────────────────────────────────────────
# The inline formula in AGP_SLR.py is 2× larger than stability_selection.py.
# With beta=0 (already fixed in §2) the formula choice has no effect (both
# select D_b≤0). To make the 2× inflation visible this section uses beta=0.05
# for both runs, holding all other settings at Python defaults, so the reader
# can see which lambda each formula selects.
# Note: combined with the §2 fix (beta=0) this cause is absorbed.
###############################################################################

def section3_normalization(seed: int = 42):
    print("\n" + "=" * 70)
    print("SECTION 3 — Fix instability normalization  [CAUSE #2: 2× inflation]")
    print("NOTE: effect is isolated here at beta=0.05; absorbed by §2 at beta=0.")
    print("=" * 70)

    n_cases = len(CASES)
    fig, axes = plt.subplots(n_cases, 4, figsize=(20, 5 * n_cases))
    axes = np.atleast_2d(axes)

    for ri, case in enumerate(CASES):
        print(f"\n  [{case}]")
        se   = load_se_solution(case)
        raw  = load_counts(case)
        clr  = clr_python(raw)
        S    = empirical_cov(clr, bias=True)
        p    = S.shape[0]
        grid = LAMBDA_GRID[case]

        # both at beta=0.05 to expose formula difference
        res_inline  = run_stars(clr.values, grid, n_sub=N_SUB, beta=0.05,
                                instability_fn=_instability_inline,  seed=seed)
        res_correct = run_stars(clr.values, grid, n_sub=N_SUB, beta=0.05,
                                instability_fn=_instability_correct, seed=seed)

        sol_inline  = refit(S, res_inline["lambda_star"])
        sol_correct = refit(S, res_correct["lambda_star"])

        frob_before = frobenius(sol_inline["Theta"],  se["Theta"])
        frob_after  = frobenius(sol_correct["Theta"], se["Theta"])

        print(f"  inline  formula (β=0.05) → idx={res_inline['opt_idx']}"
              f"  λ={res_inline['lambda_star']:.5f}  Frob={frob_before:.4f}")
        print(f"  correct formula (β=0.05) → idx={res_correct['opt_idx']}"
              f"  λ={res_correct['lambda_star']:.5f}  Frob={frob_after:.4f}")
        _record("instability normalization (2× inline)", "correct formula",
                case, frob_before, frob_after)

        # also show effective beta for inline formula (threshold at D_b_correct ≤ 0.025)
        print(f"  (inline with β=0.05 is equivalent to correct formula with β≈0.025)")

        ax = axes[ri]
        x = list(range(len(grid)))
        ax[0].plot(x, res_inline["D_b"],  "o-",  label="inline (2× inflated)")
        ax[0].plot(x, res_correct["D_b"], "s--", label="correct formula")
        ax[0].axhline(0.05, color="k", lw=1, linestyle="--", alpha=0.5, label="β=0.05")
        ax[0].axvline(res_inline["opt_idx"],  color="tab:blue", lw=1.5,
                      linestyle=":", label=f"λ* inline  idx={res_inline['opt_idx']}")
        ax[0].axvline(res_correct["opt_idx"], color="tab:orange", lw=1.5,
                      linestyle=":", label=f"λ* correct idx={res_correct['opt_idx']}")
        ax[0].set_xlabel("λ index"); ax[0].set_ylabel("D_b")
        ax[0].set_title(f"{case}: D_b — formula comparison (β=0.05)")
        ax[0].legend(fontsize=7)

        # ratio of the two D_b curves (should be ≈ 2 everywhere)
        ratio = res_inline["D_b"] / np.clip(res_correct["D_b"], 1e-12, None)
        ax[1].plot(x, ratio, "o-")
        ax[1].axhline(2.0, color="r", lw=1, linestyle="--", label="expected ratio=2")
        ax[1].set_xlabel("λ index"); ax[1].set_ylabel("D_b inline / D_b correct")
        ax[1].set_title(f"{case}: formula ratio"); ax[1].legend(fontsize=7)

        _heatmap(ax[2], sol_inline["Theta"] - se["Theta"],
                 f"{case}: Θ(inline)−Θ_se\nFrob={frob_before:.3f}")
        _heatmap(ax[3], sol_correct["Theta"] - se["Theta"],
                 f"{case}: Θ(correct)−Θ_se\nFrob={frob_after:.3f}")

    fig.suptitle("Section 3 — Effect of instability formula (shown at β=0.05)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "section3_normalization.png")


###############################################################################
# SECTION 4 — Fix CLR pseudocount    [CAUSE #3]
# ─────────────────────────────────────────────────────────────────────────────
# Python: multiplicative replacement (zeros → 1, then rescale column sum).
# SE:     uniform +1 to every entry.
# Accumulated fixes: beta=0, correct formula.
###############################################################################

def section4_clr_pseudocount(seed: int = 42):
    print("\n" + "=" * 70)
    print("SECTION 4 — Fix CLR pseudocount  [CAUSE #3: multiplicative vs uniform +1]")
    print("=" * 70)

    n_cases = len(CASES)
    fig, axes = plt.subplots(n_cases, 4, figsize=(20, 5 * n_cases))
    axes = np.atleast_2d(axes)

    for ri, case in enumerate(CASES):
        print(f"\n  [{case}]")
        se  = load_se_solution(case)
        raw = load_counts(case)
        p   = raw.shape[0]
        grid = LAMBDA_GRID[case]

        clr_py_df = clr_python(raw)
        clr_se_df = clr_se(raw)
        S_py = empirical_cov(clr_py_df, bias=True)
        S_se = empirical_cov(clr_se_df, bias=True)

        frob_cov = frobenius(S_py, S_se)
        print(f"  ||S_py − S_se||_F            = {frob_cov:.6f}")
        print(f"  S_py diag: [{np.diag(S_py).min():.4f}, {np.diag(S_py).max():.4f}]")
        print(f"  S_se diag: [{np.diag(S_se).min():.4f}, {np.diag(S_se).max():.4f}]")
        print(f"  getMaxCov(S_py) = {get_max_cov(S_py):.5f}")
        print(f"  getMaxCov(S_se) = {get_max_cov(S_se):.5f}")

        # accumulated fixes: beta=0, correct formula
        res_py = run_stars(clr_py_df.values, grid, n_sub=N_SUB, beta=0.0,
                           instability_fn=_instability_correct, seed=seed)
        res_se = run_stars(clr_se_df.values, grid, n_sub=N_SUB, beta=0.0,
                           instability_fn=_instability_correct, seed=seed)

        sol_py = refit(S_py, res_py["lambda_star"])
        sol_se = refit(S_se, res_se["lambda_star"])

        frob_before = frobenius(sol_py["Theta"], se["Theta"])
        frob_after  = frobenius(sol_se["Theta"], se["Theta"])

        print(f"  Python CLR → idx={res_py['opt_idx']}  λ={res_py['lambda_star']:.5f}"
              f"  Frob={frob_before:.4f}")
        print(f"  SE CLR     → idx={res_se['opt_idx']}  λ={res_se['lambda_star']:.5f}"
              f"  Frob={frob_after:.4f}")
        _record("CLR pseudocount (mult. vs uniform +1)", "uniform +1 (SE)",
                case, frob_before, frob_after)

        ax = axes[ri]
        # scatter S_py vs S_se off-diagonal
        mask_od = ~np.eye(p, dtype=bool)
        s_py_od = S_py[mask_od]
        s_se_od = S_se[mask_od]
        lim = max(np.abs(s_py_od).max(), np.abs(s_se_od).max()) * 1.1
        ax[0].scatter(s_py_od, s_se_od, alpha=0.25, s=6)
        ax[0].plot([-lim, lim], [-lim, lim], "r--", lw=1)
        ax[0].set_xlabel("S_py (mult. CLR)"); ax[0].set_ylabel("S_se (+1 CLR)")
        ax[0].set_title(f"{case}: off-diag cov scatter\n||S_py−S_se||_F={frob_cov:.4f}")

        _heatmap(ax[1], S_py - S_se, f"{case}: S_py − S_se")
        _heatmap(ax[2], sol_py["Theta"] - se["Theta"],
                 f"{case}: Θ(py-CLR)−Θ_se\nFrob={frob_before:.3f}")
        _heatmap(ax[3], sol_se["Theta"] - se["Theta"],
                 f"{case}: Θ(se-CLR)−Θ_se\nFrob={frob_after:.3f}")

    fig.suptitle("Section 4 — Effect of CLR pseudocount strategy", fontsize=11)
    fig.tight_layout()
    _save(fig, "section4_clr_pseudocount.png")


###############################################################################
# SECTION 5 — Fix lambda grid spacing    [CAUSE #5]
# ─────────────────────────────────────────────────────────────────────────────
# Python: log-spaced (from SPIEC-EASI printout, but log-spaced by inspection).
# SE:     linear-spaced (getLamPath default: log=FALSE).
# Accumulated fixes: beta=0, correct formula, SE CLR.
###############################################################################

def section5_lambda_grid(seed: int = 42):
    print("\n" + "=" * 70)
    print("SECTION 5 — Fix λ grid spacing  [CAUSE #5: log-spaced vs linear]")
    print("=" * 70)

    n_cases = len(CASES)
    fig, axes = plt.subplots(n_cases, 3, figsize=(17, 5 * n_cases))
    axes = np.atleast_2d(axes)

    for ri, case in enumerate(CASES):
        print(f"\n  [{case}]")
        se  = load_se_solution(case)
        raw = load_counts(case)
        p   = raw.shape[0]

        # accumulated: SE CLR
        clr = clr_se(raw)
        S   = empirical_cov(clr, bias=True)

        grid_log    = LAMBDA_GRID[case]
        grid_linear = linear_lambda_grid(S, n_lambda=20, min_ratio=1e-2)

        print(f"  Log    grid: [{grid_log[0]:.4f} … {grid_log[-1]:.4f}]")
        print(f"  Linear grid: [{grid_linear[0]:.4f} … {grid_linear[-1]:.4f}]")

        res_log = run_stars(clr.values, grid_log,    n_sub=N_SUB, beta=0.0,
                            instability_fn=_instability_correct, seed=seed)
        res_lin = run_stars(clr.values, grid_linear, n_sub=N_SUB, beta=0.0,
                            instability_fn=_instability_correct, seed=seed)

        sol_log = refit(S, res_log["lambda_star"])
        sol_lin = refit(S, res_lin["lambda_star"])

        frob_before = frobenius(sol_log["Theta"], se["Theta"])
        frob_after  = frobenius(sol_lin["Theta"], se["Theta"])

        print(f"  Log-grid   → idx={res_log['opt_idx']}  λ={res_log['lambda_star']:.5f}"
              f"  Frob={frob_before:.4f}")
        print(f"  Linear-grid → idx={res_lin['opt_idx']}  λ={res_lin['lambda_star']:.5f}"
              f"  Frob={frob_after:.4f}")
        _record("λ grid spacing (log vs linear)", "linear (SE getLamPath)",
                case, frob_before, frob_after)

        ax = axes[ri]
        idx_x = list(range(20))
        ax[0].plot(idx_x, grid_log,    "o-",  label="log-spaced (Python)")
        ax[0].plot(idx_x, grid_linear, "s--", label="linear-spaced (SE)")
        ax[0].axvline(res_log["opt_idx"], color="tab:blue", lw=1.5,
                      linestyle=":", label=f"λ* log idx={res_log['opt_idx']}")
        ax[0].axvline(res_lin["opt_idx"], color="tab:red",  lw=1.5,
                      linestyle=":", label=f"λ* lin idx={res_lin['opt_idx']}")
        ax[0].set_xlabel("grid index"); ax[0].set_ylabel("λ value")
        ax[0].set_title(f"{case}: grid comparison"); ax[0].legend(fontsize=7)

        _heatmap(ax[1], sol_log["Theta"] - se["Theta"],
                 f"{case}: Θ(log-grid)−Θ_se\nFrob={frob_before:.3f}")
        _heatmap(ax[2], sol_lin["Theta"] - se["Theta"],
                 f"{case}: Θ(lin-grid)−Θ_se\nFrob={frob_after:.3f}")

    fig.suptitle("Section 5 — Effect of λ grid spacing", fontsize=11)
    fig.tight_layout()
    _save(fig, "section5_lambda_grid.png")


###############################################################################
# SECTION 6 — Fix covariance bias    [CAUSE #6]
# ─────────────────────────────────────────────────────────────────────────────
# Python: np.cov(bias=True)  → divides by n.
# SE (R): cov()              → divides by n-1.
# Scale difference = n/(n-1). The linear lambda grid is recomputed from each
# covariance to keep the grid-to-matrix scale ratio consistent.
# Accumulated fixes: SE CLR, beta=0, correct formula, linear grid.
###############################################################################

def section6_cov_bias(seed: int = 42):
    print("\n" + "=" * 70)
    print("SECTION 6 — Fix covariance bias  [CAUSE #6: 1/n vs 1/(n-1)]")
    print("=" * 70)

    n_cases = len(CASES)
    fig, axes = plt.subplots(n_cases, 3, figsize=(17, 5 * n_cases))
    axes = np.atleast_2d(axes)

    for ri, case in enumerate(CASES):
        print(f"\n  [{case}]")
        se  = load_se_solution(case)
        raw = load_counts(case)
        p, n = raw.shape

        clr = clr_se(raw)
        S_biased   = empirical_cov(clr, bias=True)
        S_unbiased = empirical_cov(clr, bias=False)

        expected_ratio = n / (n - 1)
        actual_ratio   = get_max_cov(S_unbiased) / get_max_cov(S_biased)
        print(f"  n={n}  theoretical S ratio = {expected_ratio:.6f}")
        print(f"  actual λ_max ratio        = {actual_ratio:.6f}")
        print(f"  ||S_biased − S_unbiased||_F = {frobenius(S_biased, S_unbiased):.6f}")

        # recompute linear grid from each covariance so that λ/λ_max is the same
        grid_b  = linear_lambda_grid(S_biased)
        grid_ub = linear_lambda_grid(S_unbiased)

        res_b  = run_stars(clr.values, grid_b,  n_sub=N_SUB, beta=0.0,
                           instability_fn=_instability_correct, seed=seed)
        res_ub = run_stars(clr.values, grid_ub, n_sub=N_SUB, beta=0.0,
                           instability_fn=_instability_correct, seed=seed)

        sol_b  = refit(S_biased,   res_b["lambda_star"])
        sol_ub = refit(S_unbiased, res_ub["lambda_star"])

        frob_before = frobenius(sol_b["Theta"],  se["Theta"])
        frob_after  = frobenius(sol_ub["Theta"], se["Theta"])

        print(f"  1/n    → idx={res_b['opt_idx']}  λ={res_b['lambda_star']:.5f}"
              f"  Frob={frob_before:.4f}")
        print(f"  1/(n-1) → idx={res_ub['opt_idx']}  λ={res_ub['lambda_star']:.5f}"
              f"  Frob={frob_after:.4f}")
        _record("covariance bias (1/n vs 1/(n-1))", "bias=False (1/(n-1))",
                case, frob_before, frob_after)

        ax = axes[ri]
        mask_od = ~np.eye(p, dtype=bool)
        lim = max(np.abs(S_biased[mask_od]).max(),
                  np.abs(S_unbiased[mask_od]).max()) * 1.1
        ax[0].scatter(S_biased[mask_od], S_unbiased[mask_od], alpha=0.25, s=6)
        ax[0].plot([-lim, lim], [-lim, lim], "r--", lw=1)
        ax[0].set_xlabel("S (1/n)"); ax[0].set_ylabel("S (1/(n-1))")
        ax[0].set_title(f"{case}: covariance scatter (off-diag)\nratio≈{expected_ratio:.4f}")

        _heatmap(ax[1], sol_b["Theta"] - se["Theta"],
                 f"{case}: Θ(1/n)−Θ_se\nFrob={frob_before:.3f}")
        _heatmap(ax[2], sol_ub["Theta"] - se["Theta"],
                 f"{case}: Θ(1/(n-1))−Θ_se\nFrob={frob_after:.3f}")

    fig.suptitle("Section 6 — Effect of covariance divisor (1/n vs 1/(n-1))",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "section6_cov_bias.png")


###############################################################################
# SECTION 7 — Cumulative fix validation
# ─────────────────────────────────────────────────────────────────────────────
# Applies ALL fixes simultaneously:
#   • beta=0                         (§2)
#   • correct instability formula    (§3)
#   • SE-style CLR (uniform +1)      (§4)
#   • linear lambda grid             (§5)
#   • covariance bias=False (1/(n-1)) (§6)
#
# Reports: Frobenius norm before/after, Jaccard edge overlap,
#          and whether the residual divergence reaches the §1 solver lower bound.
###############################################################################

def section7_cumulative(seed: int = 42):
    print("\n" + "=" * 70)
    print("SECTION 7 — Cumulative fix validation (all fixes applied together)")
    print("=" * 70)

    n_cases = len(CASES)
    fig, axes = plt.subplots(n_cases, 5, figsize=(26, 5 * n_cases))
    axes = np.atleast_2d(axes)

    for ri, case in enumerate(CASES):
        print(f"\n  [{case}]")
        se  = load_se_solution(case)
        raw = load_counts(case)
        p   = raw.shape[0]

        # ── ORIGINAL (all unfixed) ────────────────────────────────────────────
        clr_orig = clr_python(raw)
        S_orig   = empirical_cov(clr_orig, bias=True)
        res_orig = run_stars(clr_orig.values, LAMBDA_GRID[case],
                             n_sub=N_SUB, beta=0.05,
                             instability_fn=_instability_inline, seed=seed)
        sol_orig = refit(S_orig, res_orig["lambda_star"])

        # ── ALL FIXES APPLIED ─────────────────────────────────────────────────
        clr_fixed = clr_se(raw)                              # fix 3: SE CLR
        S_fixed   = empirical_cov(clr_fixed, bias=False)    # fix 6: unbiased
        grid_fixed = linear_lambda_grid(S_fixed)             # fix 5: linear grid
        res_fixed = run_stars(clr_fixed.values, grid_fixed,
                              n_sub=N_SUB, beta=0.0,         # fix 1: beta=0
                              instability_fn=_instability_correct,  # fix 2
                              seed=seed)
        sol_fixed = refit(S_fixed, res_fixed["lambda_star"])

        # ── SOLVER LOWER BOUND: sweep all λ on fixed S ───────────────────────
        frob_lb = np.inf
        for lam in grid_fixed:
            ft = frobenius(refit(S_fixed, lam)["Theta"], se["Theta"])
            if ft < frob_lb:
                frob_lb = ft

        frob_orig  = frobenius(sol_orig["Theta"],  se["Theta"])
        frob_fixed = frobenius(sol_fixed["Theta"], se["Theta"])
        frob_omega_orig  = frobenius(sol_orig["Omega"],  se["Omega"])
        frob_omega_fixed = frobenius(sol_fixed["Omega"], se["Omega"])
        jac_orig  = jaccard_edges(sol_orig["Theta"],  se["Theta"])
        jac_fixed = jaccard_edges(sol_fixed["Theta"], se["Theta"])

        print(f"  ORIGINAL:  ||Θ−Θ_se||_F={frob_orig:.4f}  "
              f"||Ω−Ω_se||_F={frob_omega_orig:.4f}  Jaccard={jac_orig:.4f}")
        print(f"  ALL FIXED: ||Θ−Θ_se||_F={frob_fixed:.4f}  "
              f"||Ω−Ω_se||_F={frob_omega_fixed:.4f}  Jaccard={jac_fixed:.4f}")
        print(f"  Solver lower bound (best-λ sweep on fixed S): {frob_lb:.4f}")
        gap = frob_fixed - frob_lb
        print(f"  Remaining gap above lower bound: {gap:.4f}"
              f"  {'✓ solver-level only' if gap < 0.1 * frob_orig else '✗ residual unexplained causes'}")

        _record("ALL FIXES combined", "cumulative (§2–§6)", case,
                frob_orig, frob_fixed)

        ax = axes[ri]
        # [0] D_b curves: original vs fixed
        x_orig  = list(range(len(LAMBDA_GRID[case])))
        x_fixed = list(range(len(grid_fixed)))
        ax[0].plot(x_orig,  res_orig["D_b"],  "o-",  label="original")
        ax[0].plot(x_fixed, res_fixed["D_b"], "s--", label="all fixed")
        ax[0].axvline(res_orig["opt_idx"],  color="tab:blue", lw=1.5,
                      linestyle=":", label=f"λ* orig idx={res_orig['opt_idx']}")
        ax[0].axvline(res_fixed["opt_idx"], color="tab:red",  lw=1.5,
                      linestyle=":", label=f"λ* fixed idx={res_fixed['opt_idx']}")
        ax[0].set_xlabel("λ index"); ax[0].set_ylabel("D_b")
        ax[0].set_title(f"{case}: D_b — original vs all-fixed")
        ax[0].legend(fontsize=7)

        _heatmap(ax[1], sol_orig["Theta"] - se["Theta"],
                 f"{case}: Θ_orig−Θ_se\nFrob={frob_orig:.3f}")
        _heatmap(ax[2], sol_fixed["Theta"] - se["Theta"],
                 f"{case}: Θ_fixed−Θ_se\nFrob={frob_fixed:.3f}")
        _heatmap(ax[3], sol_orig["Omega"] - se["Omega"],
                 f"{case}: Ω_orig−Ω_se\nFrob={frob_omega_orig:.3f}")
        _heatmap(ax[4], sol_fixed["Omega"] - se["Omega"],
                 f"{case}: Ω_fixed−Ω_se\nFrob={frob_omega_fixed:.3f}")

    fig.suptitle("Section 7 — Cumulative validation: original vs all fixes applied",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "section7_cumulative.png")


###############################################################################
# ── Summary table ─────────────────────────────────────────────────────────────
###############################################################################

def print_summary():
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    if not _SUMMARY:
        print("  (no data; run sections first)")
        return

    df = pd.DataFrame(_SUMMARY, columns=[
        "cause", "fix", "case", "frob_before", "frob_after", "delta"
    ])
    print(df.to_string(index=False))

    out_path = os.path.join(PROJECT_ROOT, "analysis", "summary_table.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved summary → {out_path}")


###############################################################################
# ── Dispatch ──────────────────────────────────────────────────────────────────
###############################################################################

SECTIONS = {
    1: section1_baseline,
    2: section2_stars_beta,
    3: section3_normalization,
    4: section4_clr_pseudocount,
    5: section5_lambda_grid,
    6: section6_cov_bias,
    7: section7_cumulative,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 SLR diagnostic comparison")
    parser.add_argument(
        "--section", type=int, default=None,
        help="Run a single section (1–7). Omit to run all sections.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for StARS subsampling (default: 42).",
    )
    args = parser.parse_args()

    # always run from project root so relative data paths resolve
    os.chdir(PROJECT_ROOT)

    if args.section is not None:
        if args.section not in SECTIONS:
            print(f"Unknown section {args.section}. Choose 1–7.")
            sys.exit(1)
        fn = SECTIONS[args.section]
        # section1 has no seed arg; others do
        if args.section == 1:
            fn()
        else:
            fn(seed=args.seed)
    else:
        section1_baseline()
        for n in range(2, 8):
            SECTIONS[n](seed=args.seed)

    print_summary()
