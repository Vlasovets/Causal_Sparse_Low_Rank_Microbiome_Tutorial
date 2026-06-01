#!/usr/bin/env Rscript
# Sparse graphical lasso NetCoMi plots — AGP (family level)
#
# Fixes the incorrect edge count in the old netcomi_sparse_*.png plots,
# which loaded Alice's SLR Theta. This script loads the Python sparse GL
# precision matrices (py_sparse_theta_smoker/non_smoker.csv) which have
# 24 (smoker) and 41 (non-smoker) edges at lambda* = 0.707 / 0.537.
#
# Output -> results/two_group/figures/
#   netcomi_sparse_smoker.png / .svg
#   netcomi_sparse_non_smoker.png / .svg

suppressPackageStartupMessages({ library(NetCoMi) })

args        <- commandArgs(trailingOnly = FALSE)
script_file <- sub("--file=", "", args[grep("--file=", args)])
ROOT        <- normalizePath(file.path(dirname(script_file), ".."))
source(file.path(ROOT, "analysis", "phylum_palette.R"), local=TRUE)

SRC_AG  <- file.path(ROOT, "source", "design_AG")
SPARSE  <- file.path(ROOT, "results", "two_group", "sparse")
FIG_DIR <- file.path(ROOT, "results", "two_group", "figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Load sparse GL precision matrices (Python results) ─────────────────────
load_theta <- function(path) {
  m <- read.csv(path, row.names = 1, check.names = TRUE)
  colnames(m) <- sub("^X", "", colnames(m))
  as.matrix(m)
}

theta_sm <- load_theta(file.path(SPARSE, "py_sparse_theta_smoker.csv"))
theta_ns <- load_theta(file.path(SPARSE, "py_sparse_theta_non_smoker.csv"))

# ── Partial correlations ────────────────────────────────────────────────────
theta_to_pcor <- function(Theta) {
  d    <- sqrt(diag(Theta))
  P    <- -Theta / outer(d, d)
  diag(P) <- 0
  P[abs(P) < 1e-10] <- 0
  P
}
pcor_sm <- theta_to_pcor(theta_sm)
pcor_ns <- theta_to_pcor(theta_ns)

n_sm <- sum(pcor_sm[upper.tri(pcor_sm)] != 0)
n_ns <- sum(pcor_ns[upper.tri(pcor_ns)] != 0)
message(sprintf("Sparse edges: smoker=%d, non-smoker=%d", n_sm, n_ns))

# ── Taxonomy & labels ───────────────────────────────────────────────────────
taxa_ids <- colnames(theta_sm)
tax      <- read.csv(file.path(SRC_AG, "tax_table_smoker.csv"),
                     row.names = 1, stringsAsFactors = FALSE)

make_label <- function(id) {
  fam <- tax[id, "Family"]
  if (!is.na(fam) && nchar(fam) > 1 && fam != "nan") return(fam)
  ord <- tax[id, "Order"]
  if (!is.na(ord) && nchar(ord) > 1) return(paste0("[", ord, "]"))
  return(paste0("OTU_", id))
}
labels <- setNames(sapply(taxa_ids, make_label), taxa_ids)

phyla <- setNames(sapply(taxa_ids, function(id) {
  ph <- tax[id, "Phylum"]
  if (is.na(ph) || ph == "") "Unknown" else ph
}), labels[taxa_ids])

# Rename matrix rows/cols to labels
rownames(pcor_sm) <- colnames(pcor_sm) <- labels[taxa_ids]
rownames(pcor_ns) <- colnames(pcor_ns) <- labels[taxa_ids]

# Use shared fixed palette (phylum_palette.R) for cross-dataset colour consistency
node_cols <- setNames(sapply(phyla, phylum_colour), names(phyla))
message(sprintf("AGP phyla: %s", paste(sort(unique(phyla)), collapse=", ")))

# ── NetCoMi ─────────────────────────────────────────────────────────────────
message("Constructing networks ...")
net <- netConstruct(data     = pcor_sm,
                    data2    = pcor_ns,
                    dataType = "condDependence",
                    sparsMethod = "none",
                    normMethod  = "none",
                    verbose     = 0,
                    seed        = 123456)

props <- netAnalyze(net, clustMethod = "cluster_fast_greedy", verbose = FALSE)

# ── Shared layout from the denser (non-smoker) network ──────────────────────
p_ref <- plot(props,
              groupNames = c("Smoker", "Non-Smoker"),
              sameLayout = TRUE,
              rmSingles  = FALSE,
              nodeColor  = "colorVec",
              colorVec   = node_cols,
              featVecCol = phyla,
              legendArgs = list(title = "Phylum"),
              repulsion  = 0.9,
              labelScale = FALSE,
              cexLabels  = 0.70,
              labelCol   = "black")
layout_ref <- p_ref$layout$layout1

# ── Load permApprox significant edges (FDR < 0.1) ────────────────────────────
perm_file <- file.path(SPARSE, "sparse_edge_pvals.csv")
sig_pairs <- data.frame(taxon_i=character(), taxon_j=character(), stringsAsFactors=FALSE)
if (file.exists(perm_file)) {
  perm_df  <- read.csv(perm_file, stringsAsFactors=FALSE)
  # Map OTU IDs in perm file back to family labels
  perm_df$label_i <- labels[as.character(perm_df$taxon_i)]
  perm_df$label_j <- labels[as.character(perm_df$taxon_j)]
  sig_rows <- perm_df[perm_df$bh_pval < 0.1, c("label_i","label_j")]
  sig_rows <- sig_rows[!is.na(sig_rows$label_i) & !is.na(sig_rows$label_j), ]
  sig_pairs <- data.frame(taxon_i=sig_rows$label_i, taxon_j=sig_rows$label_j,
                           stringsAsFactors=FALSE)
  message(sprintf("permApprox significant edges (FDR<0.1): %d", nrow(sig_pairs)))
}

make_edge_colors <- function(pcor_mat, sig_pairs,
                              col_pos_sig="#E84646", col_neg_sig="#0072B2",
                              col_pos="#CCCCCC",     col_neg="#BBBBBB",
                              thresh=1e-10) {
  taxa <- rownames(pcor_mat)
  n    <- length(taxa)
  cols <- c()
  for (i in seq_len(n-1)) {
    for (j in seq(i+1, n)) {
      v <- pcor_mat[i,j]
      if (abs(v) <= thresh) next
      ti <- taxa[i]; tj <- taxa[j]
      is_sig <- any((sig_pairs$taxon_i==ti & sig_pairs$taxon_j==tj) |
                    (sig_pairs$taxon_i==tj & sig_pairs$taxon_j==ti))
      cols <- c(cols, if (v > 0) (if (is_sig) col_pos_sig else col_pos)
                      else        (if (is_sig) col_neg_sig else col_neg))
    }
  }
  cols
}

# ── Save combined two-group plot and individual cropped panels ───────────────
save_combined <- function(stem, group1, group2) {
  for (ext in c("png", "svg")) {
    out <- file.path(FIG_DIR, paste0(stem, ".", ext))
    if (ext == "png") png(out, width = 4800, height = 2000, res = 300)
    else               svg(out, width = 16, height = 6.67)
    plot(props,
         groupNames = c(paste0("Smoker  |  ", n_sm, " edges"),
                        paste0("Non-Smoker  |  ", n_ns, " edges")),
         sameLayout = TRUE,
         layout     = layout_ref,
         rmSingles  = FALSE,
         nodeColor  = "colorVec",
         colorVec   = node_cols,
         featVecCol = phyla,
         repulsion  = 0.9,
         labelScale = FALSE,
         cexLabels  = 0.70)
    dev.off()
    message(sprintf("  Saved: %s.%s", stem, ext))
  }
}

# Individual panels: plot two-group but crop to left or right half via margins
save_panel <- function(stem, which_group, group_label, n_edges) {
  for (ext in c("png", "svg")) {
    out <- file.path(FIG_DIR, paste0(stem, ".", ext))
    if (ext == "png") png(out, width = 2400, height = 2000, res = 300)
    else               svg(out, width = 8, height = 6.67)

    gnames <- if (which_group == 1)
      c(paste0("Sparse graphical lasso — ", group_label, "  |  ", n_edges, " edges"), "")
    else
      c("", paste0("Sparse graphical lasso — ", group_label, "  |  ", n_edges, " edges"))

    # Use layout from the relevant group
    lay <- if (which_group == 1) layout_ref else p_ref$layout$layout2

    plot(props,
         groupNames = gnames,
         sameLayout = FALSE,
         layout     = lay,
         rmSingles  = FALSE,
         nodeColor  = "colorVec",
         colorVec   = node_cols,
         featVecCol = phyla,
         repulsion  = 0.9,
         labelScale = FALSE,
         cexLabels  = 0.70,
         # Hide the unwanted panel by making it blank
         mar        = if (which_group == 1) c(2,2,4,2) else c(2,2,4,2))
    dev.off()
    message(sprintf("  Saved: %s.%s", stem, ext))
  }
}

message("Saving AGP sparse network plots ...")
save_combined("netcomi_sparse_agp_combined", "Smoker", "Non-Smoker")

# Individual panels via single-group netConstruct (workaround: pass both groups, mask titles)
for (grp in list(list(stem="netcomi_sparse_smoker",     pcor=pcor_sm, label="Smoker",     n=n_sm),
                 list(stem="netcomi_sparse_non_smoker",  pcor=pcor_ns, label="Non-Smoker", n=n_ns))) {
  for (ext in c("png", "svg")) {
    out <- file.path(FIG_DIR, paste0(grp$stem, ".", ext))
    if (ext == "png") png(out, width = 2400, height = 2000, res = 300)
    else               svg(out, width = 8, height = 6.67)
    net1 <- netConstruct(data = grp$pcor, data2 = grp$pcor,
                         dataType = "condDependence",
                         sparsMethod = "none", normMethod = "none",
                         verbose = 0, seed = 123456)
    pr1  <- netAnalyze(net1, clustMethod = "cluster_fast_greedy", verbose = FALSE)
    plot(pr1,
         groupNames = c(paste0("Sparse graphical lasso — ", grp$label,
                                "  |  ", grp$n, " edges"), ""),
         sameLayout = TRUE,
         layout     = layout_ref,
         rmSingles  = FALSE,
         nodeColor  = "colorVec",
         colorVec   = node_cols,
         featVecCol = phyla,
         repulsion  = 0.9,
         labelScale = FALSE,
         cexLabels  = 0.70)
    dev.off()
    message(sprintf("  Saved: %s.%s", grp$stem, ext))
  }
}

# ── Differential network (same shared layout) ────────────────────────────────
message("Computing AGP sparse differential network ...")
diff_net <- diffnet(net, diffMethod = "fisher", n1 = 234, n2 = 234)
n_diff   <- sum(diff_net$diffMat != 0, na.rm = TRUE)
message(sprintf("  Differential edges: %d", n_diff))

if (n_diff > 0) {
  diff_edge_col <- c("#CC79A7","#009E73","#0072B2","#E69F00",
                     "#999999","#56B4E9","#F0E442","white","#B55E00")
  for (ext in c("png", "svg")) {
    out <- file.path(FIG_DIR, paste0("netcomi_sparse_diff_agp.", ext))
    if (ext == "png") png(out, width = 2800, height = 2400, res = 300)
    else               svg(out, width = 9.33, height = 8)
    plot(diff_net,
         layout     = layout_ref,
         rmSingles  = FALSE,
         mar        = c(2, 2, 5, 10),
         edgeCol    = diff_edge_col,
         edgeWidth  = 2.5,
         labelScale = FALSE,
         cexLabels  = 0.70,
         title1     = paste0("Differential sparse network (AGP)  |  ",
                              n_diff, " differing edges"))
    dev.off()
    message(sprintf("  Saved: netcomi_sparse_diff_agp.%s", ext))
  }
} else {
  message("  No significant differential associations — skipping diff plot.")
}

message("AGP sparse NetCoMi plots done.")
