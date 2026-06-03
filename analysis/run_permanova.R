#!/usr/bin/env Rscript
# permANOVA of smoker vs non-smoker microbiome composition
# following Christian's permApprox + adonis2 workflow.
#
# Design:
#   adonis2(otu ~ W + sex + age + bmi, method="bray",
#           permutations=how(nperm=5000, blocks=pair_nb))
#   + perm_approx(method="gamma") for precise p-values
#
# Two distance metrics:
#   1. Bray-Curtis on raw counts     (comparable to Alice's analysis)
#   2. Euclidean on CLR counts       (Aitchison; comparable to our SLR)
#
# Outputs (results/two_group/permanova/):
#   permanova_results.csv
#   permanova_pval_hist_bray.png
#   permanova_pval_hist_aitchison.png

suppressPackageStartupMessages({
  library(vegan)
  library(permute)
  library(permApprox)
})

# ── Paths ──────────────────────────────────────────────────────────────────
args        <- commandArgs(trailingOnly = FALSE)
script_file <- sub("--file=", "", args[grep("--file=", args)])
root        <- normalizePath(file.path(dirname(script_file), ".."))
src_dir     <- file.path(root, "source", "design_AG")
ag_dir      <- file.path(root, "..", "Causal_Microbiome_Tutorial", "design_AG")
out_dir     <- file.path(root, "results", "two_group", "permanova")
fig_dir     <- file.path(root, "results", "two_group", "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

N_PERM <- 5000

# ── Load OTU tables ────────────────────────────────────────────────────────
message("Loading data ...")
otu_sm <- read.csv(file.path(src_dir, "otu_table_smoker.csv"),
                   row.names = 1, check.names = FALSE)
otu_ns <- read.csv(file.path(src_dir, "otu_table_non_smoker.csv"),
                   row.names = 1, check.names = FALSE)
otu    <- rbind(otu_sm, otu_ns)

meta_sm <- read.csv(file.path(src_dir, "sample_data_smoker.csv"),
                    row.names = 1, stringsAsFactors = FALSE)
meta_ns <- read.csv(file.path(src_dir, "sample_data_non_smoker.csv"),
                    row.names = 1, stringsAsFactors = FALSE)
meta    <- rbind(meta_sm, meta_ns)

# Add age and BMI from full AGP metadata
ag <- read.csv(file.path(ag_dir, "agdata_smoke.csv"),
               row.names = 1, check.names = FALSE)
meta$age <- ag[rownames(meta), "age_corrected"]
meta$bmi <- ag[rownames(meta), "bmi"]

# Align
otu  <- otu[rownames(meta), ]
meta$W      <- factor(meta$W,   levels = c(0, 1), labels = c("smoker", "non_smoker"))
meta$sex    <- factor(meta$sex)
meta$pair_nb <- factor(meta$pair_nb)

n <- nrow(otu)
p <- ncol(otu)
message(sprintf("n=%d samples, p=%d taxa", n, p))

# Drop samples with missing age or BMI
keep <- !is.na(meta$age) & !is.na(meta$bmi)
otu  <- otu[keep, ]
meta <- meta[keep, ]
message(sprintf("After dropping NA age/bmi: n=%d", nrow(meta)))

# ── CLR transform ──────────────────────────────────────────────────────────
clr_transform <- function(counts) {
  x <- log(counts + 1)
  x - rowMeans(x)
}
clr <- clr_transform(otu)

# ── Permutation design: shuffle within matched pairs ───────────────────────
perm_ctrl <- how(
  nperm  = N_PERM,
  blocks = meta$pair_nb
)

# ── Helper: run adonis2 + permApprox ──────────────────────────────────────
run_analysis <- function(data_mat, dist_method, label) {
  message(sprintf("\n=== %s (%s distance) ===", label, dist_method))

  set.seed(42)
  adon <- adonis2(
    data_mat ~ W + sex + age + bmi,
    data         = meta,
    method       = dist_method,
    by           = "terms",
    permutations = perm_ctrl
  )
  message("adonis2 result:")
  print(adon)

  ps         <- permustats(adon)
  obs_stats  <- as.numeric(ps$statistic)
  perm_stats <- as.matrix(ps$permutations)
  names(obs_stats)     <- names(ps$statistic)
  colnames(perm_stats) <- names(ps$statistic)

  gpd_ctrl <- make_gpd_ctrl(
    constraint  = "support_at_obs",
    sample_size = nrow(data_mat)
  )

  pa <- perm_approx(
    obs_stats   = obs_stats,
    perm_stats  = perm_stats,
    method      = "gpd",
    null_center = "mean",
    gpd_ctrl    = gpd_ctrl,
    verbose     = FALSE
  )

  out <- data.frame(
    distance    = dist_method,
    term        = names(obs_stats),
    F_obs       = round(obs_stats, 4),
    R2          = round(adon$R2[seq_along(obs_stats)], 4),
    adonis_p    = adon$`Pr(>F)`[seq_along(obs_stats)],
    empirical_p = pa$p_empirical,
    approx_p    = pa$p_unadjusted,
    approx_p_BH = pa$p_values,
    fit_status  = pa$fit_result$status,
    stringsAsFactors = FALSE
  )

  message("\nResults:")
  print(out)

  # Histogram of permutation null for W (smoking)
  p_w <- out$approx_p[out$term == "W"]
  draw_null_hist <- function() {
    hist(perm_stats[, "W"], breaks = 100, col = "#555555", border = "white",
         main = sprintf("permANOVA null distribution — smoking (W)\n%s distance, n=%d, %d permutations",
                        dist_method, nrow(data_mat), N_PERM),
         xlab = "Pseudo-F statistic")
    abline(v = obs_stats["W"], col = "#C0392B", lwd = 2.5)
    legend("topright",
           legend = sprintf("Observed F = %.3f\np_approx = %.2e",
                            obs_stats["W"], p_w),
           bty = "n", text.col = "#C0392B")
  }
  png(file.path(fig_dir, sprintf("permanova_null_%s.png", dist_method)),
      width = 1200, height = 700, res = 150)
  draw_null_hist(); dev.off()
  svg(file.path(fig_dir, sprintf("permanova_null_%s.svg", dist_method)),
      width = 8, height = 4.67)
  draw_null_hist(); dev.off()
  message(sprintf("Saved: permanova_null_%s.{{png,svg}}", dist_method))

  out
}

# ── Run both distance metrics ──────────────────────────────────────────────
res_bray      <- run_analysis(otu,  "bray",      "Bray-Curtis")
res_aitchison <- run_analysis(clr,  "euclidean", "Aitchison (CLR + Euclidean)")

# ── Save combined results ──────────────────────────────────────────────────
results <- rbind(res_bray, res_aitchison)
out_csv <- file.path(out_dir, "permanova_results.csv")
write.csv(results, out_csv, row.names = FALSE)
message(sprintf("\nSaved: %s", out_csv))
message("\nDone.")
