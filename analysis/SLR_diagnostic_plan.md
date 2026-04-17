# SLR Diagnostic Plan: Python (GGLasso) vs. SPIEC-EASI

**Date:** 2026-04-06  
**Implementations compared:**
- Python: `notebooks/AGP_SLR.py`, `utils/solver.py`, `utils/stability_selection.py`
- R reference: `source/5_networks_AG/SE_SLR.Rmd` (SpiecEasi `method='slr'`)

---

## 1. Optimization Objective

Both implementations minimize the **latent-variable Gaussian graphical lasso**:

```
min_{Ω, Θ, L}  −log det(Ω) + Tr(S·Ω) + λ‖Θ‖₁,ₒd + μ₁‖L‖★
s.t.  Ω = Θ − L,  Ω ≻ 0,  L ≽ 0
```

When `r` is specified (always the case here, `r=10`), the nuclear norm penalty on L is
replaced by a **hard rank-r constraint** via a rank-truncated eigendecomposition. Both
implementations use the same proximal operator: threshold at the (r+1)-th largest
eigenvalue and zero out everything below it. The Python code explicitly cites the
SpiecEasi C++ source at this line (`utils/solver.py:318`).

**Conclusion:** objectives are equivalent given the same `S`, `λ`, and `r`.

---

## 2. Algorithm and Stopping Criteria

| Aspect | Python (GGLasso ADMM) | SPIEC-EASI C++ ADMM |
|---|---|---|
| Solver | Scaled ADMM, adaptive ρ (Boyd et al.) | Same paper, C++ backend |
| ρ update rule | ×2 if r≥10s, ×0.5 if s≥10r (`solver.py:179–188`) | Similar heuristic |
| Stopping tolerances | `tol=1e-5`, `rtol=1e-5` (`AGP_SLR.py:316`) | Unknown; likely 1e-4 default |
| Max iterations | 1000 | Unknown (often 100 in pulsar) |
| Convergence criterion | Boyd primal+dual residuals | Boyd primal+dual residuals |
| Initialization | `Omega_0 = I_p`, `Theta_0 = I_p`, `X_0 = 0` | Unknown |

**Update order (Python):**
```
W_t  = Θ_t − L_t − X_t − (1/ρ)S
Ω_t  = phiplus(W_t, β=1/ρ)            # project onto S++
Θ_t  = prox_l1_offdiag(Ω_t+L_t+X_t)  # soft-threshold off-diagonal
L_t  = prox_rank_r(Θ_t − X_t − Ω_t)  # hard rank-r truncation
X_t  = X_t + Ω_t − Θ_t + L_t          # scaled dual update
```

**Divergence probability from solver differences: LOW** — both use the same ADMM
formulation. Tighter tolerances in Python (1e-5 vs likely 1e-4 in SE) mean Python
solutions may be slightly more accurate.

---

## 3. Input Data Preprocessing

### 3.1 CLR Transformation Pipeline (Revised — Primary Suspect #3)

The user note and SPIEC-EASI source confirm that **both** implementations pass the
CLR covariance (not the correlation matrix) to the ADMM optimizer. However, the
two CLR pipelines differ in their pseudocount strategy:

**SPIEC-EASI R** (`spiec-easi.R`, `.spiec.easi.norm`):
```r
.spiec.easi.norm <- function(data) {
  t(clr(data + 1, 1))   # add +1 to ALL entries, then CLR, then transpose
}
```
- Adds a **uniform pseudocount of +1 to every entry**, including already-nonzero counts.
- For a taxon with 1000 counts, the effective value becomes 1001 (+0.1%).
- For a zero entry, the effective value becomes 1.
- CLR is then applied to each sample (row of the n×p transposed matrix).

**Python** (`utils/helper.py`, `transform_features`):
```python
X = zero_imputation(X, pseudo_count=1)   # replace zeros with 1, then rescale column sums
X = normalize(X)                          # divide by column sum → relative abundances
X = log_transform(X, "clr")              # log(X / geomean_per_sample(X))
```
`zero_imputation` uses **multiplicative replacement**: zeros become 1, then every
column is scaled by `original_sum / shifted_sum` to preserve the total count.
Nonzero entries are scaled down slightly; zero entries become a fractional 1.

| Property | R (uniform +1) | Python (multiplicative replacement) |
|---|---|---|
| Effect on zeros | becomes 1 | becomes `1 × scale ≈ 1/(1 + n_zeros)` |
| Effect on nonzeros | count+1 (slightly inflated) | count × scale (slightly deflated) |
| Geometric mean shift | all components shifted up | only zeros replaced |

For typical microbiome data with 30–70% zeros per sample, these approaches produce
**meaningfully different CLR values**, especially for taxa with many zeros. The
geometric mean used in CLR changes differently, which propagates to the covariance
matrix passed to the ADMM solver.

### 3.2 Covariance Matrix (Primary Suspect #1 — Revised)

Both implementations compute the empirical covariance of CLR-transformed data, so
the "covariance vs. correlation" framing from the initial analysis is **incorrect**.
The actual difference is:

| | R | Python |
|---|---|---|
| Estimator | `cov(X)`, unbiased, divisor `1/(n-1)` | `np.cov(bias=True)`, divisor `1/n` |
| Matrix orientation | X is n×p before `cov()` | `clr_counts` is p×n before `np.cov()` |
| Result | `S_R = (1/(n-1)) ΣΣ` | `S_py = (1/n) ΣΣ` |
| Scale ratio | `S_R = n/(n-1) × S_py` | — |

For n=100: S_R is ~1.01× S_py (1% larger).  
For n=50: S_R is ~1.02× S_py (2% larger).

**Lambda grid alignment:** Python reuses SPIEC-EASI's printed lambda values directly
(`AGP_SLR.py:308–313`). Those values were computed by SE as
`getMaxCov(cov(X_R))` = `max_offdiag(|S_R|)`. Python applies those same lambda
values against `S_py`, which is `(n-1)/n × S_R`. The effective lambda relative to
the input covariance scale is therefore slightly larger in Python — minor
over-regularization of ~1–2% for large n.

**Conclusion:** 1/(n-1) vs 1/n is a **minor second-order effect** when λ values are
shared between implementations. The primary consequence of different preprocessing
comes from the pseudocount strategy (§3.1), not the covariance divisor.

---

## 4. Hyperparameters

| Parameter | Python | SPIEC-EASI R | Equivalent? |
|---|---|---|---|
| λ grid | Reused from SE output (`AGP_SLR.py:308–313`) | 20 log-spaced values, `lambda.min.ratio=1e-2` | Same numeric values |
| λ grid spacing | Log-spaced (ratio ≈ 1.27 per step) | `getLamPath(..., log=FALSE)` → **linear** spacing by default | **DIFFERENT** |
| r (rank) | Swept 2–10; plotted at `selected_rank=10` | Fixed `r=10` | Same at r=10 |
| μ₁ | `mu1=1` (`AGP_SLR.py:318`), irrelevant when r is given | N/A (hard rank constraint) | N/A |
| StARS β threshold | `beta=0.05` (`AGP_SLR.py:317`) | `beta=0` (`SE_SLR.Rmd:25`) | **NOT equivalent** |
| StARS subsamples | N=10 (`AGP_SLR.py:63`) | `rep.num=20` (`SE_SLR.Rmd:25`) | Different |
| Random seed | None set | Not shown | Stochastic divergence |
| ADMM tol | 1e-5 / 1e-5 | Unknown SE default | Likely different |

### 4.1 Lambda Grid Spacing: Log vs. Linear

`getLamPath` in the `pulsar` package signature is:
```r
getLamPath <- function(max, min, len, log=FALSE) {
  lams <- seq(max, min, length.out=len)  # LINEAR by default
  ...
}
```
SpiecEasi's call does not explicitly pass `log=TRUE`, so its grid is **linearly
spaced** from λ_max down to λ_max × 0.01. Python's grid uses
`np.logspace(0, -2, 10)` (or the hardcoded values which are log-spaced by
inspection: ratio ≈ 1.27 per step). This means the grids, while having the same
endpoints and same number of points, place different density around intermediate
λ values. The optimal λ index from StARS therefore picks a different absolute
λ value even if the same index is selected.

### 4.2 StARS Beta: 0 vs. 0.05 (Primary Suspect #1)

This is the single most impactful difference. The `pulsar` package's StARS
implementation selects the **largest λ where total instability D_b ≤ β**.

- Python (`beta=0.05`): selects a λ that allows moderate edge instability — the
  standard StARS regime, typically producing a connected sparse graph.
- SPIEC-EASI (`beta=0`): selects the largest λ where D_b ≤ 0, meaning
  **perfect stability** is required. This forces selection of the most
  regularized feasible λ, producing a sparser or empty graph.

These two criteria will almost always select **different λ indices** from the
same path, even if the path itself were identical. The resulting Θ matrices will
have different sparsity levels and different element magnitudes.

---

## 5. Model Selection: StARS Instability Normalization Discrepancy

Two formulas appear in the Python codebase and they differ by a factor of 2:

**Inline StARS in `AGP_SLR.py:385`:**
```python
total_instability = 2 * np.sum(edge_instability) / (p * (p - 1))
```
Sums all p(p-1) off-diagonal elements of the symmetric instability matrix (each
edge counted twice), divides by p(p-1)/2 = C(p,2). Result: average instability
per unique edge, range [0, 2].

**`utils/stability_selection.py:72`:**
```python
total_instability = np.sum(edge_instability, axis=(0,1)) / comb(p, 2) / 2
```
Sums all p² elements (including diagonal, which may be nonzero), divides by
C(p,2)×2 = p(p-1). For a symmetric matrix with zero diagonal this equals
sum_offdiag / (p(p-1)) = half of the inline formula.

The inline formula in `AGP_SLR.py` overestimates D_b by 2× relative to the
utility function. With `beta=0.05`, the inflated D_b causes StARS to select a
**smaller λ** (more edges, less regularization) than intended, compounding the
divergence from §4.2.

---

## 6. Output Representation

| Variable | Python | SPIEC-EASI R | Same scale? |
|---|---|---|---|
| Sparse part | `sol['Theta']` (p×p) | `se$est$icov[[opt_idx]]` | Yes, both are Θ |
| Low-rank part | `sol['L']` (p×p, PSD) | `se$est$resid[[opt_idx]]` | Yes, both are L |
| Precision matrix | `sol['Omega']` = Θ−L | Computed as `se_Theta − se_L` in Python (`AGP_SLR.py:279`) | Yes |

Both use the decomposition Ω = Θ − L, where Θ is sparse and L is low-rank PSD.
The precision matrix Ω is the object used for likelihood and eBIC.

**Caution:** `$est$resid` in SpiecEasi stores the low-rank residual L, not the
fitting residual Ω−S. Verify this against the SpiecEasi C++ output struct if
results remain inconsistent after addressing the above suspects.

---

## 7. Ranked Root Causes of Divergence

| Rank | Cause | Mechanism | Probability |
|---|---|---|---|
| **1** | **StARS beta=0.05 vs. beta=0** | Python's threshold allows D_b ≤ 0.05; SE requires D_b = 0. Always selects different λ* → different sparsity level → different Θ, L | **Very high** |
| **2** | **StARS instability normalization (2× factor)** | Python's inline formula inflates D_b by 2×, biasing λ* toward smaller values (denser graphs). Compounds Cause #1 | **High** |
| **3** | **CLR pseudocount strategy** | R adds +1 uniformly to all counts; Python uses multiplicative replacement for zeros only. For sparse count tables (>50% zeros) this produces different covariance matrices S passed to the ADMM | **High** |
| **4** | **N=10 vs. N=20 subsamples + no random seed** | Fewer subsamples → noisier D_b estimates → less reliable λ* selection. No seed means results are not reproducible across runs | **Medium** |
| **5** | **Lambda grid spacing: log vs. linear** | Same λ_max and λ_min, but different density in between. Same StARS-selected index maps to different absolute λ | **Medium** |
| **6** | **Covariance divisor 1/n vs. 1/(n-1)** | SE's S is n/(n-1) × Python's S. Lambda values calibrated for SE applied to Python's smaller S → ~1–2% over-regularization for n=100 | **Low** |
| **7** | **ADMM convergence tolerance** | Python uses tol=1e-5, SE default likely 1e-4. Minor difference in solution accuracy | **Low** |

---

## 8. Normalization Function Summary (`.spiec.easi.norm`)

Confirmed from SpiecEasi source (`R/spiec-easi.R`):

```r
.spiec.easi.norm <- function(data) {
  if (inherits(data, 'matrix')) {
    return(t(clr(data + 1, 1)))   # pseudocount=1 (all entries), CLR, transpose
  } else if (inherits(data, 'list')) {
    return(do.call('cbind', lapply(data, .spiec.easi.norm)))
  }
}
```

For the `slr` method specifically:
```r
X <- .spiec.easi.norm(data)
if (is.null(args[['lambda.max']]))
  args$lambda.max <- getMaxCov(cov(X))   # uses R's cov() = 1/(n-1)
```

`getMaxCov` (from `pulsar`):
```r
getMaxCov <- function(x, cov=isSymmetric(x), abs=TRUE, diag=FALSE) {
  if (!cov) x <- cov(x)
  tmp <- Matrix::triu(x, k=1)        # upper triangular, excluding diagonal
  tmp <- abs(tmp)
  max(tmp)                            # max absolute off-diagonal entry
}
```

So `lambda_max = max_offdiag(|cov_R(CLR(data+1))|)` where `cov_R` uses `1/(n-1)`.

---

## 9. Phase 2 Implementation Plan

Phase 2 will be implemented in a separate file
`analysis/SLR_comparison.py`. The analysis will proceed in two stages:

### Stage A — Isolate solver differences (bypass StARS)
- Fix `r=10` throughout.
- For each λ index in the shared grid (0..19), run both solvers on the **same S**
  (Python's covariance) and compare `||Θ_py − Θ_se||_F` and `||L_py − L_se||_F`.
- This tests whether the ADMM implementations are truly equivalent given identical
  inputs.

### Stage B — Full end-to-end comparison at each implementation's own λ*
- Python: StARS with N=10, beta=0.05, inline formula.
- SE: loaded from CSV outputs (already run with N=20, beta=0).
- Compare selected λ indices, resulting Θ and L matrices.
- Quantitative metrics:
  - Frobenius norm: `||Θ_py − Θ_se||_F`, `||L_py − L_se||_F`, `||Ω_py − Ω_se||_F`
  - Sparsity pattern: Jaccard similarity of edge sets (off-diagonal nonzeros)
  - Eigenvalue spectra of L: compare top-r eigenvalues
  - Element-wise differences: heatmaps of `Θ_py − Θ_se`, `L_py − L_se`
  - eBIC at each λ: confirm whether the two selected λ* agree on the eBIC-optimal point

### Stage C — Controlled experiments to confirm root causes
1. **Fix beta=0.05 in Python, rerun**: does λ* align with SE index?
2. **Use bias=False in np.cov**: does Frobenius norm drop?
3. **Match pseudocount strategy**: apply uniform +1 to all counts before CLR.
4. **Fix N=20, set seed**: does instability estimate stabilize?
5. **Test log vs. linear lambda grid**: recompute SE-equivalent linear grid.

Each experiment produces a Frobenius norm table to quantify contribution of each cause.
