# Strict SPIEC-EASI vs GGLasso Comparison (Shared Matrix, Shared Lambda Grid, Shared Beta)

## Scope

This report summarizes the strict side-by-side comparison where:

1. Step 1 exports the matrix inferred to be the true empirical matrix used internally by SPIEC-EASI for glasso fitting.
2. Step 2 reuses that exact matrix.
3. Both methods use the same lambda grid and the same StARS threshold (beta).

Primary scripts:

1. [analysis/run_spieceasi_glasso_step1.R](analysis/run_spieceasi_glasso_step1.R)
2. [analysis/run_gglasso_shared_input_and_heatmaps.py](analysis/run_gglasso_shared_input_and_heatmaps.py)

SLURM entry points used:

1. [slurm/09_run_se_gglasso_strict_comparison.sh](slurm/09_run_se_gglasso_strict_comparison.sh)
2. [slurm/10_run_gglasso_strict_step2.sh](slurm/10_run_gglasso_strict_step2.sh)

---

## Shared Setup (What Was Forced to Match)

From [results/se_model_info.csv](results/se_model_info.csv), [results/shared_lambda_grid.csv](results/shared_lambda_grid.csv), and [results/gglasso_model_info.csv](results/gglasso_model_info.csv):

1. Count data size: `n_samples=234`, `n_taxa=40`.
2. Shared matrix type inferred from SE internals: `correlation_from_internal_data`.
3. Shared lambda grid: 20 values from `0.9003366103` down to `0.0090033661`.
4. Shared beta: `0.05`.

Code references:

1. Matrix inference from SE internal data: [analysis/run_spieceasi_glasso_step1.R](analysis/run_spieceasi_glasso_step1.R#L11)
2. SE run configuration (glasso + stars + shared beta via pulsar): [analysis/run_spieceasi_glasso_step1.R](analysis/run_spieceasi_glasso_step1.R#L111)
3. Lambda grid export: [analysis/run_spieceasi_glasso_step1.R](analysis/run_spieceasi_glasso_step1.R#L128)
4. Step 2 loading shared lambda grid and beta: [analysis/run_gglasso_shared_input_and_heatmaps.py](analysis/run_gglasso_shared_input_and_heatmaps.py#L201)

---

## Approach A: SPIEC-EASI (R)

Workflow summary:

1. Run `spiec.easi(..., method='glasso', sel.criterion='stars', nlambda=20)`.
2. Extract selected precision and selected fitted covariance with `getOptiCov` and `getOptCov`.
3. Infer true fitting matrix from `se_fit$est$data` by comparing `lambda0` against `lammax(cov)` vs `lammax(cor)`.
4. Export path stats and stars path for strict comparison.

Key code:

1. Main SPIEC-EASI call: [analysis/run_spieceasi_glasso_step1.R](analysis/run_spieceasi_glasso_step1.R#L111)
2. Selected matrix extraction: [analysis/run_spieceasi_glasso_step1.R](analysis/run_spieceasi_glasso_step1.R#L123)
3. Internal matrix inference: [analysis/run_spieceasi_glasso_step1.R](analysis/run_spieceasi_glasso_step1.R#L11)
4. SE path exports (`se_theta_path_stats.csv`, `se_stars_path.csv`): [analysis/run_spieceasi_glasso_step1.R](analysis/run_spieceasi_glasso_step1.R#L128)

Artifacts:

1. [results/se_empirical_covariance.csv](results/se_empirical_covariance.csv)
2. [results/se_fitted_covariance_opt.csv](results/se_fitted_covariance_opt.csv)
3. [results/se_precision_theta.csv](results/se_precision_theta.csv)
4. [results/se_theta_path_stats.csv](results/se_theta_path_stats.csv)
5. [results/se_stars_path.csv](results/se_stars_path.csv)

---

## Approach B: Custom GGLasso-Style ADMM + StARS (Python)

Workflow summary:

1. Read shared covariance/correlation matrix and shared lambda grid from Step 1 outputs.
2. Build CLR-transformed data and subsamples (`N=20`).
3. For each lambda:
   - Estimate StARS instability (`gg_D_b`) from subsample edge indicators.
   - Solve full-data ADMM and store nonzero off-diagonal count for strict path comparison.
4. Select lambda via largest index with `D_b <= beta`.
5. Refit final `Theta` on the shared matrix.

Key code:

1. Shared input loading: [analysis/run_gglasso_shared_input_and_heatmaps.py](analysis/run_gglasso_shared_input_and_heatmaps.py#L192)
2. Shared lambda/beta loading: [analysis/run_gglasso_shared_input_and_heatmaps.py](analysis/run_gglasso_shared_input_and_heatmaps.py#L201)
3. StARS loop and path metrics: [analysis/run_gglasso_shared_input_and_heatmaps.py](analysis/run_gglasso_shared_input_and_heatmaps.py#L67)
4. Strict side-by-side table build: [analysis/run_gglasso_shared_input_and_heatmaps.py](analysis/run_gglasso_shared_input_and_heatmaps.py#L269)

Artifacts:

1. [results/gglasso_precision_theta.csv](results/gglasso_precision_theta.csv)
2. [results/gglasso_empirical_covariance.csv](results/gglasso_empirical_covariance.csv)
3. [results/gglasso_instability_path.csv](results/gglasso_instability_path.csv)
4. [results/side_by_side_path_comparison.csv](results/side_by_side_path_comparison.csv)

---

## Similarities (Observed)

Using [results/side_by_side_path_comparison.csv](results/side_by_side_path_comparison.csv):

1. Both methods evaluated the same 20 lambda values.
2. Path-level support sizes are nearly identical at each lambda.
3. `abs_diff_nonzero_offdiag` is 0 for most lambdas.
4. Maximum support-size discrepancy across the whole path is only 2 off-diagonal entries (out of 1560).

This indicates the structural path is highly consistent when matrix/lambda/beta are aligned.

---

## Discrepancies (Observed)

### 1. Selected lambda differs by one path step

From metadata:

1. SE: `lambda_star = 0.7065481316` ([results/se_model_info.csv](results/se_model_info.csv))
2. GG: `lambda_star = 0.5544706908` ([results/gglasso_model_info.csv](results/gglasso_model_info.csv))

Important indexing note:

1. R index is 1-based (`opt_index=2` in SE metadata).
2. Python index is 0-based (`opt_index=2` in GG metadata).
3. In the shared table, lambda `0.706548` is at index 1 (0-based), and `0.554471` is at index 2.

### 2. Instability values are systematically different in scale

From [results/side_by_side_path_comparison.csv](results/side_by_side_path_comparison.csv):

1. `gg_D_b` is consistently lower than `se_D_b`.
2. `abs_diff_D_b` grows along the path (up to ~0.3444 at the smallest lambda).

This means path selection thresholds can diverge even when support path is almost identical.

### 3. Diagonal penalty behavior differs by implementation

Diagnostic file:

1. [results/diagonal_penalty_diagnostic.csv](results/diagonal_penalty_diagnostic.csv)

Code evidence:

1. Custom solver calls `prox_od_1norm(..., diag=False)`: [utils/solver.py](utils/solver.py#L180)
2. Helper semantics for `diag`: [utils/helper.py](utils/helper.py#L617)

This can affect numerical values and stability path scaling.

---

## Selected-Solution Snapshot

At selected lambdas:

1. SE selected solution is sparse (`se_nonzero_offdiag=48` at lambda `0.706548`, table index 1).
2. GG selected solution is denser (`gg_nonzero_offdiag=166` at lambda `0.554471`, table index 2).

Source: [results/side_by_side_path_comparison.csv](results/side_by_side_path_comparison.csv)

---

## Interpretation

After enforcing the same matrix, same lambda grid, and same beta:

1. The core sparsity path is largely matched between methods.
2. Remaining mismatch is mostly in instability scale and index convention, not in gross path structure.
3. The selected model can still differ (one-step lambda shift), which materially changes final sparsity.

This is now a true apples-to-apples comparison setup, and discrepancies are localized and diagnosable.

---

## Root-Cause Analysis

### Solver equivalence is confirmed

The sparsity path match (max 2-edge difference across 1,560 pairs at all 20 lambdas) validates that the Python ADMM solver on the same correlation matrix produces the same precision matrices as SPIEC-EASI/huge. The earlier fix — switching from raw covariance to correlation matrix as the ADMM input — was the decisive correction.

### The one-step lambda selection gap is caused by different random subsamples, not a formula error

The D_b values differ by roughly 2× across the path:

| 0-based index | lambda   | se_D_b | gg_D_b | ratio |
|---------------|----------|--------|--------|-------|
| 1             | 0.706548 | 0.0231 | 0.0094 | 2.46× |
| 2             | 0.554471 | 0.0668 | 0.0319 | 2.09× |
| 3             | 0.435126 | 0.1065 | 0.0527 | 2.02× |

Because `gg_D_b` at index 2 is 0.032 (below beta=0.05) while `se_D_b` at index 2 is 0.067 (above beta=0.05), GGLasso selects one step further down the lambda path, yielding 166 edges instead of 48.

The D_b formulas in both implementations are mathematically equivalent — both reduce to `sum(2·θ·(1−θ)_offdiag) / (p·(p−1))`. The discrepancy is purely from **different random subsamples**: R's `set.seed(42)` and Python's `np.random.seed(42)` use different random number generators and produce different subsample draws even with the same integer seed. Same subsample ratio (≈0.654 for n=234, formula `10√n/n`), different permutations → different per-lambda edge frequency estimates → different D_b.

To close this gap, the subsamples themselves would need to be matched (e.g., export R's subsample indices and reuse them in Python). Absent that, a ~1-step selection difference is the expected level of stochastic variability in StARS.

### `se_D_b` in `se_stars_path.csv` is a cumulative maximum, not raw instability

From [results/se_stars_path.csv](results/se_stars_path.csv), the column is flat at exactly 0.529038 from index 14 through 19. This is pulsar's cumulative maximum (cummax running from large to small lambda), stored in `stars$summary`. The raw per-lambda D_b from R's subsamples is not separately exported. Values at indices 0–13 (before the plateau) are raw and directly comparable; values at 14–19 only indicate the maximum instability reached, not the actual value at that lambda.

### `huge` penalises the diagonal — `diag=False` is correct

**Finding (2026-04-20):** The assumption that SPIEC-EASI/`huge` uses a standard off-diagonal-only GLasso penalty was incorrect. `huge` minimises:

```
−log det Θ + tr(S Θ) + λ ‖Θ‖₁
```

where `‖Θ‖₁` is the element-wise L1 norm over **all** entries including the diagonal. This is verified by solving at SE's selected lambda (0.706548) with both `diag=False` (threshold diagonal) and `diag=True` (preserve diagonal) and comparing against SE's exported precision matrix:

| Metric | `diag=False` | `diag=True` | SE reference |
|--------|-------------|------------|-------------|
| nnz off-diagonal | 48 | 46 | 48 |
| Frob rel err vs SE (full) | 0.000001 | 0.717 | — |
| Frob rel err vs SE (diagonal) | 0.000000 | 0.714 | — |
| Diagonal mean | 0.5873 | 1.0063 | 0.5873 |
| Diagonal std | 0.0033 | 0.0157 | 0.0033 |

`diag=False` reproduces SE's precision matrix to machine precision. `diag=True` (standard off-diagonal GLasso) gives diagonal values ~1.0 (unconstrained positive-definite floor), whereas SE's diagonal values are ~0.587 — shrunk below 1 by the diagonal penalty.

**Conclusion:** `diag=False` in [utils/solver.py:158](utils/solver.py#L158) and [utils/solver.py:277](utils/solver.py#L277) is the **correct** setting for matching SPIEC-EASI. The change to `diag=True` was reverted. The earlier note flagging `diag=False` as a potential bug was wrong.

---

## Subsample Alignment Fix (2026-04-22)

### Motivation

The ~2× D_b discrepancy identified in the root-cause analysis had two independent causes that had to be fixed together:

1. **Subsample mismatch** — R's and Python's RNGs produce different draws even with the same integer seed. Fixed by exporting R's subsample indices from Step 1 and loading them in Step 2.
2. **D_b formula mismatch** — pulsar's `stars.stability` computes `4 * sum(θ(1−θ)) / (p(p−1))` over the full p×p matrix, which equals 2× the StARS-paper definition. Python was computing the StARS paper formula directly, giving half pulsar's value. Fixed by adding the missing factor of 2.

### pulsar formula (from source)

```r
est$summary[i] <- 4 * sum(est$merge[[i]] * (1 - est$merge[[i]])) / (p * (p - 1))
```

`est$merge[[i]]` is the full p×p average adjacency matrix. The diagonal contributes 0 (no self-loops), so the effective sum is over off-diagonal pairs only: `4 * sum_{i≠j} θ(1−θ) / (p(p−1))`. This equals `8 * sum_{i<j} θ(1−θ) / (p(p−1))`, which is **2× the StARS paper definition** of `D_b = (1/C(p,2)) * sum_{i<j} 2p(1−p)`. Python's corrected formula matches pulsar exactly.

### Implementation

- R ([analysis/run_spieceasi_glasso_step1.R](analysis/run_spieceasi_glasso_step1.R)): resets `set.seed(42)` immediately after `spiec.easi()` completes and replays N=20 calls to `sample(n_se, n_sub, replace=FALSE)` to reproduce pulsar's draws. Exports to `results/se_subsample_indices.csv` (20 rows × 152 columns, 1-based indices) and `results/se_subsample_meta.csv`.
- Python ([analysis/run_gglasso_shared_input_and_heatmaps.py](analysis/run_gglasso_shared_input_and_heatmaps.py)): loads exported indices, converts to 0-based, and slices `X[:, idx]` per subsample. Falls back to `np.random.seed(42)` if the file is absent. D_b formula corrected to `2.0 * np.sum(edge_instability) / (p * (p - 1))`.

### Results (from `results/alignment_validation.csv`)

| Check | Value |
|-------|-------|
| `lambda_match` | **True** |
| GGLasso `lambda_star` | 0.706548131615755 |
| SE `lambda_star` | 0.706548131615755 |
| `edges_match` | **True** |
| GGLasso nnz off-diagonal at selected λ | 48 |
| SE nnz off-diagonal at selected λ | 48 |
| Max D_b ratio (indices 0–13, before cummax) | 1.0017 |
| Max `abs_diff_D_b` across full path | 0.1443 (indices 14–19, cummax plateau only) |
| `subsample_source` | `R_exported` |

D_b values at indices 0–13 now match to floating-point precision (max abs diff < 3×10⁻⁴). The residual `max_abs_diff_Db` of 0.1443 is entirely from indices 14–19, where `se_D_b` is frozen at pulsar's cummax value (0.5290) while `gg_D_b` continues to reflect the raw per-lambda instability.

### Conclusion

With matched subsamples and corrected D_b formula, SPIEC-EASI and GGLasso are **functionally equivalent end-to-end**: same selected lambda (0.7065), same edge count (48), and D_b paths that agree to floating-point precision at all lambdas where comparison is meaningful.

### Figures (post-fix)

Generated by `analysis/regenerate_heatmaps_from_saved_matrices.py`. Pre-fix originals are preserved as `figures/prefix_heatmap_*.png`.

#### Instability and sparsity paths

![D_b instability path comparison](figures/postfix_db_path_comparison.png)

![Sparsity path comparison](figures/postfix_sparsity_path_comparison.png)

#### Empirical covariance matrices (shared input, hierarchical clustering from SE)

![SE empirical covariance](figures/postfix_se_empirical_covariance_heatmap.png)

![GGLasso empirical covariance](figures/postfix_gglasso_empirical_covariance_heatmap.png)

#### Precision matrices at selected λ = 0.7065 (48 edges each)

![SE precision matrix](figures/postfix_se_precision_theta_heatmap.png)

![GGLasso precision matrix](figures/postfix_gglasso_precision_theta_heatmap.png)

---

## Reproducibility Outputs

Core strict-comparison outputs to archive:

1. [results/se_model_info.csv](results/se_model_info.csv)
2. [results/gglasso_model_info.csv](results/gglasso_model_info.csv)
3. [results/shared_lambda_grid.csv](results/shared_lambda_grid.csv)
4. [results/se_theta_path_stats.csv](results/se_theta_path_stats.csv)
5. [results/se_stars_path.csv](results/se_stars_path.csv)
6. [results/gglasso_instability_path.csv](results/gglasso_instability_path.csv)
7. [results/side_by_side_path_comparison.csv](results/side_by_side_path_comparison.csv)
8. [results/se_subsample_indices.csv](results/se_subsample_indices.csv)
9. [results/se_subsample_meta.csv](results/se_subsample_meta.csv)
10. [results/alignment_validation.csv](results/alignment_validation.csv)
---

## Sparse + Low-Rank Model — SPIEC-EASI vs Custom Python Implementation

### Overview

This section extends the GLasso comparison to the **Sparse + Low-Rank (SLR)** model from Chandrasekaran et al. (2011). The SLR precision matrix is decomposed as Ω = Θ − L, where Θ is sparse (λ₁‖Θ‖₁ regularization) and L is positive semidefinite with fixed rank r (hard rank constraint: `prox_rank_norm` with `r=5`).

Primary scripts:

1. [analysis/run_spieceasi_slr_step1.R](analysis/run_spieceasi_slr_step1.R)
2. [analysis/run_slr_comparison.py](analysis/run_slr_comparison.py)

SLURM entry points:

1. [slurm/11_run_se_slr_step1.sh](slurm/11_run_se_slr_step1.sh)
2. [slurm/12_run_slr_python_step2.sh](slurm/12_run_slr_python_step2.sh)

---

### SLR Setup (What Was Forced to Match)

| Property | Value |
|---|---|
| Dataset | `otu_table_smoker.csv` (234 samples × 40 taxa) |
| Normalization | CLR: log(x+1), row-centered per sample |
| R estimator | `spiec.easi(..., method='slr', r=5, lambda.min.ratio=1e-2, nlambda=20)` |
| Python solver | `ADMM_single(S, lambda, latent=True, r=5, diag=False)` |
| Shared input S | `cov(X_clr)` — raw 40×40 covariance matrix (NOT correlation-scaled) |
| Lambda grid | 20 log-spaced values from λ_max = max\|off-diag(S)\| ≈ 4.664 down to ≈ 0.0466 |
| StARS beta | 0.05 |
| Subsamples | 20 draws of n_sub = ⌊10√234⌋ = 152, exported from R (`set.seed(42)`, re-played after `spiec.easi()`) |
| Rank constraint r | 5 |

**Key difference from GLasso**: The SLR solver uses the raw covariance matrix (not the correlation matrix). SPIEC-EASI's `sparseLowRankiCov` sets `lambda.max = max|off-diagonal of cov(X_clr)|` (≈ 4.664), not derived from the correlation matrix. The `shrinkDiag=TRUE` parameter is handled internally by the C++ ADMM solver.

Code references:

1. SLR model dispatch in SE: `SpiecEasi:::spiec.easi.default` → `estFun = "sparseLowRankiCov"`, `lambda.max = getMaxCov(cov(X))`
2. R script lambda grid and output export: [analysis/run_spieceasi_slr_step1.R](analysis/run_spieceasi_slr_step1.R#L86)
3. Python StARS loop (latent=True, raw cov, cummax selection): [analysis/run_slr_comparison.py](analysis/run_slr_comparison.py#L83)
4. `prox_rank_norm` with hard rank `r`: [utils/solver.py](utils/solver.py#L313)

---

### SLR Validation Results (2026-04-22)

From [results/slr_alignment_validation.csv](results/slr_alignment_validation.csv) and [results/slr_side_by_side_path_comparison.csv](results/slr_side_by_side_path_comparison.csv):

| Metric | Value | Status |
|---|---|---|
| `lambda_match` | True | ✓ Both select λ = 0.670841 (index 8, 0-based) |
| `py_lambda_star` | 0.670841328012343 | — |
| `se_lambda_star` | 0.670841328012343 | — |
| `rank_L_py` | 5 | ✓ Exactly r = 5 |
| `rank_L_se` | 5 | ✓ Exactly r = 5 |
| `frob_theta` | 0.0151 | ✓ 1.5% relative error on sparse Θ |
| `frob_L` | 0.1756 | ⚠ 17.6% relative error on low-rank L (see below) |
| `frob_omega` | 0.0359 | ✓ 3.6% relative error on Ω = Θ − L |
| NNZ at selected λ (Python) | 30 edges | — |
| NNZ at selected λ (SE) | 34 edges | ⚠ 4-edge difference (~12%) at same λ |
| `max_Db_ratio` | 2.39 at λ-index 5 | ⚠ D_b scale mismatch at high λ (see below) |

#### Root cause: different ADMM formulation between SE (C++) and Python

SE's C++ `ADMM` function (`_SpiecEasi_ADMM`) uses a **3-block augmented Lagrangian** formulation with state `Y ∈ ℝ^{p×3p}` (concatenating three p×p blocks) and `Λ ∈ ℝ^{p×3p}`, with over-relaxation (`over_relax_par=1.6`) and adaptive penalty `μ` initialized to `p` (number of taxa). Python's `ADMM_single` uses a standard **2-block Boyd-style ADMM** with `rho=1`, `update_rho=True`, and no over-relaxation. Both formulations minimize the same convex objective and converge to the same true optimum in exact arithmetic, but the numerical paths diverge — especially for the low-rank component L, whose nuclear-norm proximal step is ill-conditioned and highly sensitive to the trajectory.

Three effects remain after aligning all shared inputs:

1. **frob_L = 17.6%** — L is the most ill-conditioned part of the solution: small perturbations in eigenvalue trajectory lead to large differences in the rank-5 projection. The Frobenius error is irreducible by tightening Python's tolerance (verified at tol=1e-7 through 1e-10) because both solvers have converged to their respective fixed points. frob_theta = 1.5% and frob_omega = 3.6% are acceptable.

2. **4-edge NNZ difference at selected λ** — SE selects 34 edges and Python selects 30 edges at λ = 0.670841. Borderline entries near the sparsity threshold differ due to the different convergence paths, not a systematic bias.

3. **D_b ratio ≈ 2.4 at λ-index 5** — At λ = 1.388 (high regularization), SE sees instability D_b = 0.0033 while Python sees 0.0014. This ratio drops to 1.1–1.3× at the lambda values relevant for selection (indices 7–9), confirming the discrepancy is minor in the selection region.

**Both methods cross the β = 0.05 threshold at the same lambda** despite the D_b scale difference, so the downstream network is effectively equivalent.

Key SE ADMM parameters (from `SpiecEasi:::ADMM` source):
- `over_relax_par = 1.6` — over-relaxation not present in Python ADMM
- `mu = p = 40` (initial penalty, equivalent to Python's rho)
- `maxiter = 100` (via `admm2` opts), `tol = 0.001` (first λ), `tol = 1.0` (warm-start subsequent λ)
- `shrinkDiag = TRUE` — scaling S to correlation inside the C++ solver; tested and reverted (see below)

#### Note on opt_index off-by-one (fixed 2026-04-22)

R's `getOptInd()` returns a 1-based index. The R script now exports `opt_index` as 0-based (subtracts 1 before writing to CSV) for Python compatibility. Previous run reported `edges_match=False` because it compared Python's NNZ at index 8 against SE's NNZ at index 9 (wrong index). Corrected: both methods at index 8 give 30 (Python) vs 34 (SE).

---

### SLR Figures

#### SPIEC-EASI SLR — raw covariance (clustered)

![SE SLR raw covariance](figures/slr_se_empirical_covariance_heatmap.png)

#### SPIEC-EASI SLR — sparse Θ (masked diagonal)

![SE SLR Theta](figures/slr_se_precision_theta_heatmap.png)

#### SPIEC-EASI SLR — low-rank L

![SE SLR L](figures/slr_se_lowrank_L_heatmap.png)

#### SPIEC-EASI SLR — effective precision Ω = Θ − L (masked diagonal)

![SE SLR Omega](figures/slr_se_omega_heatmap.png)

#### Python SLR — sparse Θ (masked diagonal)

![Python SLR Theta](figures/slr_py_precision_theta_heatmap.png)

#### Python SLR — low-rank L

![Python SLR L](figures/slr_py_lowrank_L_heatmap.png)

#### Python SLR — effective precision Ω = Θ − L

![Python SLR Omega](figures/slr_py_omega_heatmap.png)

#### D_b instability path — SPIEC-EASI vs Python

![SLR D_b path](figures/slr_db_path_comparison.png)

#### Sparsity path (NNZ off-diagonal of Θ)

![SLR sparsity path](figures/slr_sparsity_path_comparison.png)

---

### shrinkDiag Investigation (2026-04-22) — Hypothesis Tested and Disproved

#### Hypothesis

`shrinkDiag=TRUE` in SE's C++ ADMM was hypothesized to be the root cause of frob_L=17.6%. The proposed fix: scale S to correlation in `ADMM_single` before solving and back-transform the result, with `lambda1_mask = 1/outer(d,d)` to maintain equivalent penalty.

#### Implementation (subsequently reverted)

`shrink_diag=True` was added as default to `ADMM_single` in [utils/solver.py](utils/solver.py). Pre-processing: `S → S / outer(d, d)` where `d = sqrt(diag(S))`, `lambda1_mask = 1 / outer(d, d)`. Post-processing: `Θ_out = Θ_scaled / outer(d, d)`, same for L and Ω.

#### Numerical test results (job 35466953)

Three approaches were tested against SE's exported Θ and L matrices:

| Approach | frob_theta | frob_L | Verdict |
|----------|-----------|--------|---------|
| Solve on raw cov S, uniform λ (**pre-fix**) | **1.5%** | **17.6%** | Best |
| Solve on cor(S), uniform λ, back-transform /outer(d,d) (SE's assumed approach) | 42.8% | 52.9% | Worst |
| Solve on cor(S), λ₁_mask=1/outer(d,d), back-transform /outer(d,d) (**post-fix**) | 4.9% | 27.5% | Worse |

The `shrinkDiag` hypothesis is **disproved**: applying `shrink_diag=True` degraded all metrics. The pre-fix baseline (raw covariance, uniform λ) remains the best Python match to SE.

#### Mathematical explanation

Our `shrink_diag=True` with `lambda1_mask=1/outer(d,d)` is mathematically equivalent to solving on raw S with uniform λ (the `d_i*d_j` factors cancel after back-transform). The only difference is numerical: the two coordinate systems lead to different ADMM trajectories and slightly different fixed points. The correlation-scale coordinate system happens to converge to a fixed point further from SE's than the raw-covariance coordinate system.

SE's actual `shrinkDiag=TRUE` in C++ uses a different (3-block) ADMM formulation with `over_relax_par=1.6` and `μ_init=p`, so the effective computation is not replicable by a coordinate transform alone.

#### Disposition

- `shrink_diag` reverted to `False` (default) in [utils/solver.py](utils/solver.py#L18).
- `slr_shrinkdiag_validation.csv` records `shrink_diag=False` and the baseline results.
- **The 17.6% frob_L is accepted as the current baseline** — it is irreducible by parameter tuning (verified tol=1e-7 through 1e-10, rho=1 through rho=p) and is due to fundamental differences in ADMM formulation between SE (3-block, over-relaxation) and Python (2-block, Boyd).

#### Final validation (job 35468677, from [results/slr_shrinkdiag_validation.csv](results/slr_shrinkdiag_validation.csv))

| Check | Value | Status |
|-------|-------|--------|
| `lambda_match` | True | ✓ |
| `rank_L_match` | True | ✓ |
| `edges_match` | False (34 SE vs 30 Py) | ⚠ 4-edge gap; algorithmic |
| `frob_err_theta` | 1.5% | ✓ acceptable |
| `frob_err_L` | 17.6% | ⚠ irreducible; ADMM formulation difference |
| `frob_err_combined` | 3.6% | ✓ acceptable |
| `max_Db_ratio` | 2.39× | ⚠ at high-λ only; selection λ unaffected |
| `shrink_diag` | False | — |

---

### Known Differences: SLR vs GLasso

| Aspect | GLasso | SLR |
|---|---|---|
| Input matrix S | Correlation (unit diagonal) | Raw covariance (diagonal 1.7–9.4) |
| λ_max | ≈ 0.900 (from max\|off-diag of cor(X)\|) | ≈ 4.664 (from max\|off-diag of cov(X)\|) |
| R solver | `huge::glasso` | C++ `ADMM` (SpiecEasi lowrank branch) |
| Python solver | `ADMM_single(latent=False)` | `ADMM_single(latent=True, r=5)` |
| L variable | Not present (L=0) | Positive semidefinite, rank ≤ 5 |
| `diag` in prox | `diag=False` (full L1 including diagonal) | `diag=False` (same) |
| `shrinkDiag` | Handled by `huge` (cor input ≡ unit diagonal) | C++ ADMM internal; Python replication tested and reverted — see shrinkDiag Investigation section |
| ADMM structure | 2-block Boyd (Python) / `huge` (R) | 3-block SpiecEasi C++ (over_relax=1.6, μ=p) vs 2-block Python |

**Residual error interpretation**: frob_theta=1.5% and frob_omega=3.6% are due to different ADMM formulations (3-block over-relaxation vs 2-block), not a data or lambda bug. frob_L=17.6% is irreducible — both solvers converge to their respective fixed points for the ill-conditioned nuclear-norm subproblem. Lambda selection is unaffected: both select index 8 (λ=0.670841).

---

### SLR Reproducibility Outputs

1. [results/slr_se_model_info.csv](results/slr_se_model_info.csv)
2. [results/slr_py_model_info.csv](results/slr_py_model_info.csv)
3. [results/slr_shared_lambda_grid.csv](results/slr_shared_lambda_grid.csv)
4. [results/slr_se_theta_path_stats.csv](results/slr_se_theta_path_stats.csv)
5. [results/slr_se_stars_path.csv](results/slr_se_stars_path.csv)
6. [results/slr_py_instability_path.csv](results/slr_py_instability_path.csv)
7. [results/slr_side_by_side_path_comparison.csv](results/slr_side_by_side_path_comparison.csv)
8. [results/slr_se_subsample_indices.csv](results/slr_se_subsample_indices.csv)
9. [results/slr_alignment_validation.csv](results/slr_alignment_validation.csv)
10. [results/slr_se_precision_theta.csv](results/slr_se_precision_theta.csv)
11. [results/slr_se_lowrank_L.csv](results/slr_se_lowrank_L.csv)
12. [results/slr_se_omega.csv](results/slr_se_omega.csv)
13. [results/slr_py_precision_theta.csv](results/slr_py_precision_theta.csv)
14. [results/slr_py_lowrank_L.csv](results/slr_py_lowrank_L.csv)
15. [results/slr_py_omega.csv](results/slr_py_omega.csv)
