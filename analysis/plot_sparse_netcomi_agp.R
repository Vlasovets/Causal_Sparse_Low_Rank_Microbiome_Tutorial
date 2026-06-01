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

# Helper: reindex pcor matrix to a target set of node names (pads with 0 for missing)
reindex_to_labels <- function(pcor_mat, target_names) {
  n   <- length(target_names)
  out <- matrix(0, n, n, dimnames=list(target_names, target_names))
  common <- intersect(rownames(pcor_mat), target_names)
  out[common, common] <- pcor_mat[common, common]
  out
}
message(sprintf("AGP phyla: %s", paste(sort(unique(phyla)), collapse=", ")))


# ── Step 1: Build sparse network and extract reference layout ─────────────────
message("Constructing sparse reference network ...")
net_sp <- netConstruct(data = pcor_sm, data2 = pcor_ns,
                       dataType = "condDependence",
                       sparsMethod = "none", normMethod = "none",
                       verbose = 0, seed = 123456)
props_sp <- netAnalyze(net_sp, clustMethod = "cluster_fast_greedy", verbose = FALSE)

# Use rmSingles="inboth" to define reference taxa (nodes with >=1 edge in either group)
p_ref <- plot(props_sp,
              groupNames = c("Smoker", "Non-Smoker"),
              sameLayout = TRUE,
              rmSingles  = "inboth",
              nodeColor  = "colorVec",
              colorVec   = node_cols,
              featVecCol = phyla,
              repulsion  = 0.9,
              labelScale = FALSE,
              cexLabels  = 0.70)

reference_layout <- p_ref$layout$layout1
reference_taxa   <- rownames(reference_layout)
message(sprintf("Reference taxa (with >=1 edge in sparse): %d of %d",
                length(reference_taxa), length(names(node_cols))))

# ── Load permApprox significant edges ─────────────────────────────────────────
perm_file <- file.path(SPARSE, "sparse_edge_pvals.csv")
sig_pairs <- data.frame(taxon_i=character(), taxon_j=character(), stringsAsFactors=FALSE)
if (file.exists(perm_file)) {
  perm_df  <- read.csv(perm_file, stringsAsFactors=FALSE)
  perm_df$label_i <- labels[as.character(perm_df$taxon_i)]
  perm_df$label_j <- labels[as.character(perm_df$taxon_j)]
  sig_rows <- perm_df[perm_df$bh_pval < 0.1, c("label_i","label_j")]
  sig_rows <- sig_rows[!is.na(sig_rows$label_i) & !is.na(sig_rows$label_j), ]
  sig_pairs <- data.frame(taxon_i=sig_rows$label_i, taxon_j=sig_rows$label_j, stringsAsFactors=FALSE)
  message(sprintf("permApprox significant edges (FDR<0.1): %d", nrow(sig_pairs)))
}

# ── Step 2: Sparse combined plot — reference_taxa, reference_layout ────────────
message("Saving sparse combined network (reference taxa) ...")
pcor_sm_ref <- pcor_sm[reference_taxa, reference_taxa]
pcor_ns_ref <- pcor_ns[reference_taxa, reference_taxa]
n_sm <- sum(pcor_sm_ref[upper.tri(pcor_sm_ref)] != 0)
n_ns <- sum(pcor_ns_ref[upper.tri(pcor_ns_ref)] != 0)
message(sprintf("Sparse edges (reference taxa): smoker=%d, non-smoker=%d", n_sm, n_ns))

net_sp_ref   <- netConstruct(data=pcor_sm_ref, data2=pcor_ns_ref,
                              dataType="condDependence", sparsMethod="none",
                              normMethod="none", verbose=0, seed=123456)
props_sp_ref <- netAnalyze(net_sp_ref, clustMethod="cluster_fast_greedy", verbose=FALSE)

for (ext in c("png","svg")) {
  out <- file.path(FIG_DIR, paste0("netcomi_sparse_agp_combined.", ext))
  if (ext=="png") png(out, width=4800, height=2000, res=300)
  else            svg(out, width=16, height=6.67)
  plot(props_sp_ref,
       groupNames = c(paste0("Smoker  |  ", n_sm, " edges"),
                      paste0("Non-Smoker  |  ", n_ns, " edges")),
       layout     = reference_layout,
       sameLayout = TRUE,
       rmSingles  = FALSE,
       nodeColor  = "colorVec",
       colorVec   = node_cols[reference_taxa],
       featVecCol = phyla[reference_taxa],
       repulsion  = 0.9,
       labelScale = FALSE,
       cexLabels  = 0.70)
  dev.off()
  message(sprintf("  Saved: netcomi_sparse_agp_combined.%s", ext))
}

# ── Step 3: SLR combined plot — SAME reference_taxa and reference_layout ───────
message("\n=== SLR network (same reference_taxa + reference_layout as sparse) ===")
theta_slr_sm <- load_theta(file.path(SPARSE, "..", "slr", "py_slr_theta_smoker.csv"))
theta_slr_ns <- load_theta(file.path(SPARSE, "..", "slr", "py_slr_theta_non_smoker.csv"))
pcor_slr_sm  <- theta_to_pcor(theta_slr_sm)
pcor_slr_ns  <- theta_to_pcor(theta_slr_ns)
slr_ids <- rownames(pcor_slr_sm)
rownames(pcor_slr_sm) <- colnames(pcor_slr_sm) <- labels[slr_ids]
rownames(pcor_slr_ns) <- colnames(pcor_slr_ns) <- labels[slr_ids]

# Subset SLR to exactly the same reference_taxa
pcor_slr_sm_ref <- pcor_slr_sm[reference_taxa, reference_taxa]
pcor_slr_ns_ref <- pcor_slr_ns[reference_taxa, reference_taxa]
n_slr_sm <- sum(pcor_slr_sm_ref[upper.tri(pcor_slr_sm_ref)] != 0)
n_slr_ns <- sum(pcor_slr_ns_ref[upper.tri(pcor_slr_ns_ref)] != 0)
message(sprintf("SLR edges (reference taxa): smoker=%d, non-smoker=%d", n_slr_sm, n_slr_ns))

net_slr_ref   <- netConstruct(data=pcor_slr_sm_ref, data2=pcor_slr_ns_ref,
                               dataType="condDependence", sparsMethod="none",
                               normMethod="none", verbose=0, seed=123456)
props_slr_ref <- netAnalyze(net_slr_ref, clustMethod="cluster_fast_greedy", verbose=FALSE)

for (ext in c("png","svg")) {
  out <- file.path(FIG_DIR, paste0("netcomi_slr_agp_combined.", ext))
  if (ext=="png") png(out, width=4800, height=2000, res=300)
  else            svg(out, width=16, height=6.67)
  plot(props_slr_ref,
       groupNames = c(paste0("SLR: Smoker  |  ", n_slr_sm, " edges"),
                      paste0("SLR: Non-Smoker  |  ", n_slr_ns, " edges")),
       layout     = reference_layout,
       sameLayout = TRUE,
       rmSingles  = FALSE,
       nodeColor  = "colorVec",
       colorVec   = node_cols[reference_taxa],
       featVecCol = phyla[reference_taxa],
       repulsion  = 0.9,
       labelScale = FALSE,
       cexLabels  = 0.70)
  dev.off()
  message(sprintf("  Saved: netcomi_slr_agp_combined.%s", ext))
}

# ── Differential network (same reference layout) ──────────────────────────────
message("\nComputing AGP sparse differential network ...")
diff_net <- diffnet(net_sp_ref, diffMethod="fisher", n1=234, n2=234)
n_diff   <- sum(diff_net$diffMat != 0, na.rm=TRUE)
if (n_diff > 0) {
  diff_edge_col <- c("#CC79A7","#009E73","#0072B2","#E69F00","#999999","#56B4E9","#F0E442","white","#B55E00")
  for (ext in c("png","svg")) {
    out <- file.path(FIG_DIR, paste0("netcomi_sparse_diff_agp.", ext))
    if (ext=="png") png(out, width=2800, height=2400, res=300)
    else            svg(out, width=9.33, height=8)
    plot(diff_net, layout=reference_layout, rmSingles=FALSE,
         mar=c(2,2,5,10), edgeCol=diff_edge_col, edgeWidth=2.5,
         labelScale=FALSE, cexLabels=0.70,
         title1=paste0("Differential sparse network (AGP)  |  ", n_diff, " differing edges"))
    dev.off()
    message(sprintf("  Saved: netcomi_sparse_diff_agp.%s", ext))
  }
}

message("AGP sparse + SLR NetCoMi plots done.")
