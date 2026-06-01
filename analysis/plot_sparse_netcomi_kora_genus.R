#!/usr/bin/env Rscript
# Sparse graphical lasso NetCoMi plots — KORA FF4 genus level (188 genera)
#
# Uses py_sparse_theta_smoker/non_smoker.csv from KORA sparse_permapprox.
# For 188 taxa + ~310 edges the default layout produces overlap;
# this script uses repulsion=1.8 and cexLabels=0.45 for legibility,
# and also saves a hub-only view (top 30 by degree) for the main text.
#
# Output -> KORA_Smoking_SLR/results/genus/figures/
#   netcomi_sparse_smoker.png / .svg
#   netcomi_sparse_non_smoker.png / .svg

suppressPackageStartupMessages({ library(NetCoMi) })

args        <- commandArgs(trailingOnly = FALSE)
script_file <- sub("--file=", "", args[grep("--file=", args)])
ROOT        <- normalizePath(file.path(dirname(script_file), ".."))
KORA_ROOT   <- normalizePath(file.path(ROOT, "..", "KORA_Smoking_SLR"))

SPARSE  <- file.path(KORA_ROOT, "results", "genus", "sparse_permapprox")
TAX_F   <- file.path(KORA_ROOT, "source", "tax_table.csv")
FIG_DIR <- file.path(KORA_ROOT, "results", "genus", "figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Load sparse GL precision matrices ──────────────────────────────────────
load_theta <- function(path) {
  m <- read.csv(path, row.names = 1, check.names = FALSE)
  as.matrix(m)
}

theta_sm <- load_theta(file.path(SPARSE, "py_sparse_theta_smoker.csv"))
theta_ns <- load_theta(file.path(SPARSE, "py_sparse_theta_non_smoker.csv"))

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
message(sprintf("KORA genus sparse edges: smoker=%d, non-smoker=%d", n_sm, n_ns))

# ── Taxonomy ────────────────────────────────────────────────────────────────
taxa_names <- rownames(theta_sm)

# Use full taxonomy to map genus → phylum (KORA tax_table.csv has family-level rows,
# so we use taxonomy_clean.csv from KORA_data which has genus-level entries)
tax_full <- read.csv(
  file.path(KORA_ROOT, "..", "KORA_data", "taxonomy_clean.csv"),
  row.names = 1, stringsAsFactors = FALSE)
tax_full$genus_clean <- sub("^g__", "", sub(";.*", "", tax_full$genus))
tax_full$phylum_clean <- sub("^p__", "", sub(";.*", "", tax_full$phylum))

get_phylum <- function(name) {
  idx <- which(tax_full$genus_clean == name)
  if (length(idx) > 0) {
    ph <- unique(tax_full$phylum_clean[idx])[1]
    if (!is.na(ph) && nchar(ph) > 1) return(ph)
  }
  "Unknown"
}
phyla <- setNames(sapply(taxa_names, get_phylum), taxa_names)
message(sprintf("Phyla found: %s", paste(sort(unique(phyla)), collapse=", ")))

# Use shared fixed palette for cross-dataset colour consistency
source(file.path(ROOT, "analysis", "phylum_palette.R"), local=TRUE)
node_cols <- setNames(sapply(phyla, phylum_colour), taxa_names)

# ── Build NetCoMi networks ──────────────────────────────────────────────────
message("Constructing KORA genus sparse networks ...")
net <- netConstruct(data     = pcor_sm,
                    data2    = pcor_ns,
                    dataType = "condDependence",
                    sparsMethod = "none",
                    normMethod  = "none",
                    verbose     = 0,
                    seed        = 123456)

props <- netAnalyze(net, clustMethod = "cluster_fast_greedy", verbose = FALSE)

# Extract shared layout (larger repulsion + more iterations for dense graph)
p_ref <- plot(props,
              groupNames = c("Smoker", "Non-Smoker"),
              sameLayout = TRUE,
              rmSingles  = FALSE,
              nodeColor  = "colorVec",
              colorVec   = node_cols,
              featVecCol = phyla,
              legendArgs = list(title = "Phylum"),
              repulsion  = 1.8,
              labelScale = FALSE,
              cexLabels  = 0.45,
              labelCol   = "black")
layout_raw <- p_ref$layout$layout1
if (is.null(rownames(layout_raw))) rownames(layout_raw) <- top_nodes
layout_ref <- layout_raw

# ── Hub-only combined plot (top 30 nodes by union degree) ────────────────────
N_HUBS <- 30
message(sprintf("Computing top %d hub nodes by union degree ...", N_HUBS))

# Union adjacency: node present in either network counts
adj_union <- (abs(pcor_sm) > 0) | (abs(pcor_ns) > 0)
diag(adj_union) <- FALSE
union_degree  <- rowSums(adj_union)
top_nodes     <- names(sort(union_degree, decreasing = TRUE))[1:N_HUBS]
message("Top hub genera: ", paste(head(top_nodes, 8), collapse = ", "), " ...")

pcor_sm_hub <- pcor_sm[top_nodes, top_nodes]
pcor_ns_hub <- pcor_ns[top_nodes, top_nodes]
node_cols_hub <- node_cols[top_nodes]
phyla_hub     <- phyla[top_nodes]

n_sm_hub <- sum(pcor_sm_hub[upper.tri(pcor_sm_hub)] != 0)
n_ns_hub <- sum(pcor_ns_hub[upper.tri(pcor_ns_hub)] != 0)
message(sprintf("Hub subnetwork edges: smoker=%d, non-smoker=%d", n_sm_hub, n_ns_hub))

net_hub <- netConstruct(data     = pcor_sm_hub,
                        data2    = pcor_ns_hub,
                        dataType = "condDependence",
                        sparsMethod = "none",
                        normMethod  = "none",
                        verbose     = 0,
                        seed        = 123456)
props_hub <- netAnalyze(net_hub, clustMethod = "cluster_fast_greedy",
                        verbose = FALSE)

# ── Load permApprox significant edges (FDR < 0.1) ────────────────────────────
perm_file <- file.path(SPARSE, "sparse_perm_edge_pvals.csv")
sig_pairs <- data.frame(taxon_i=character(), taxon_j=character(), stringsAsFactors=FALSE)
if (file.exists(perm_file)) {
  perm_df  <- read.csv(perm_file, stringsAsFactors=FALSE)
  sig_rows <- perm_df[perm_df$bh_pval < 0.1, c("taxon_i","taxon_j")]
  sig_pairs <- sig_rows
  message(sprintf("permApprox significant edges (FDR<0.1): %d", nrow(sig_pairs)))
}

# Build edge-colour vector for hub subgraph: highlight permApprox-sig edges
make_edge_colors <- function(pcor_mat, sig_pairs,
                              col_pos_sig="#E84646", col_neg_sig="#0072B2",
                              col_pos="#BBBBBB",     col_neg="#AAAAAA",
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
edge_cols_sm  <- make_edge_colors(pcor_sm_hub, sig_pairs)
edge_cols_ns  <- make_edge_colors(pcor_ns_hub, sig_pairs)

message("Saving KORA genus sparse hub-only combined plot ...")
for (ext in c("png", "svg")) {
  out <- file.path(FIG_DIR, paste0("netcomi_sparse_kora_combined.", ext))
  if (ext == "png") png(out, width = 4800, height = 2400, res = 300)
  else               svg(out, width = 16, height = 8)
  plot(props_hub,
       groupNames = c(paste0("Smoker  |  top ", N_HUBS, " hubs  |  ",
                              n_sm_hub, " edges"),
                      paste0("Non-Smoker  |  top ", N_HUBS, " hubs  |  ",
                              n_ns_hub, " edges")),
       sameLayout = TRUE,
       rmSingles  = FALSE,
       nodeColor  = "colorVec",
       colorVec   = node_cols_hub,
       featVecCol = phyla_hub,
       repulsion  = 1.2,
       labelScale = FALSE,
       cexLabels  = 0.75,
       title1     = paste0("Smoker | top ", N_HUBS, " hubs | ", n_sm_hub, " edges"),
       title2     = paste0("Non-Smoker | top ", N_HUBS, " hubs | ", n_ns_hub, " edges"))
  dev.off()
  message(sprintf("  Saved: netcomi_sparse_kora_combined.%s", ext))
}

message("Saving KORA genus sparse network plots ...")
for (grp in list(list(stem="netcomi_sparse_smoker",     pcor=pcor_sm, label="Smoker",     n=n_sm),
                 list(stem="netcomi_sparse_non_smoker",  pcor=pcor_ns, label="Non-Smoker", n=n_ns))) {
  for (ext in c("png", "svg")) {
    out <- file.path(FIG_DIR, paste0(grp$stem, ".", ext))
    if (ext == "png") png(out, width = 3000, height = 2800, res = 300)
    else               svg(out, width = 10, height = 9.33)
    net1 <- netConstruct(data = grp$pcor, data2 = grp$pcor,
                         dataType = "condDependence",
                         sparsMethod = "none", normMethod = "none",
                         verbose = 0, seed = 123456)
    pr1  <- netAnalyze(net1, clustMethod = "cluster_fast_greedy", verbose = FALSE)
    plot(pr1,
         groupNames = c(paste0("KORA genus — Sparse GL — ", grp$label,
                                "  |  ", grp$n, " edges"), ""),
         sameLayout = TRUE,
         layout     = layout_ref,
         rmSingles  = FALSE,
         nodeColor  = "colorVec",
         colorVec   = node_cols,
         featVecCol = phyla,
         repulsion  = 1.8,
         labelScale = FALSE,
         cexLabels  = 0.45)
    dev.off()
    message(sprintf("  Saved: %s.%s", grp$stem, ext))
  }
}

# ── Differential network on hub subgraph (same layout) ───────────────────────
message("Computing KORA genus sparse differential network (hub subgraph) ...")
diff_net_hub <- diffnet(net_hub, diffMethod = "fisher", n1 = 236, n2 = 236)
n_diff_hub   <- sum(diff_net_hub$diffMat != 0, na.rm = TRUE)
message(sprintf("  Differential edges in hub subgraph: %d", n_diff_hub))

if (n_diff_hub > 0) {
  diff_edge_col <- c("#CC79A7","#009E73","#0072B2","#E69F00",
                     "#999999","#56B4E9","#F0E442","white","#B55E00")
  for (ext in c("png", "svg")) {
    out <- file.path(FIG_DIR, paste0("netcomi_sparse_diff_kora_hub.", ext))
    if (ext == "png") png(out, width = 3200, height = 2600, res = 300)
    else               svg(out, width = 10.67, height = 8.67)
    plot(diff_net_hub,
         rmSingles  = FALSE,
         mar        = c(2, 2, 5, 10),
         edgeCol    = diff_edge_col,
         edgeWidth  = 2.5,
         labelScale = FALSE,
         cexLabels  = 0.70,
         title1     = paste0("Differential sparse network — KORA genus (top ", N_HUBS,
                              " hubs)  |  ", n_diff_hub, " differing edges"))
    dev.off()
    message(sprintf("  Saved: netcomi_sparse_diff_kora_hub.%s", ext))
  }
} else {
  message("  No significant differential associations in hub subgraph.")
}

# ── SLR network — same top-30 hub nodes and layout ───────────────────────────
message("\n=== KORA SLR network (same top-30 hubs, same layout) ===")
slr_sm_path <- file.path(KORA_ROOT, "results", "genus", "slr", "py_slr_theta_smoker.csv")
slr_ns_path <- file.path(KORA_ROOT, "results", "genus", "slr", "py_slr_theta_non_smoker.csv")

theta_slr_sm <- as.matrix(read.csv(slr_sm_path, row.names=1, check.names=FALSE))
theta_slr_ns <- as.matrix(read.csv(slr_ns_path, row.names=1, check.names=FALSE))
pcor_slr_sm  <- theta_to_pcor(theta_slr_sm)
pcor_slr_ns  <- theta_to_pcor(theta_slr_ns)

# Subset to the same top-30 hub nodes
pcor_slr_sm_hub <- pcor_slr_sm[top_nodes, top_nodes]
pcor_slr_ns_hub <- pcor_slr_ns[top_nodes, top_nodes]
n_slr_sm_hub <- sum(pcor_slr_sm_hub[upper.tri(pcor_slr_sm_hub)] != 0)
n_slr_ns_hub <- sum(pcor_slr_ns_hub[upper.tri(pcor_slr_ns_hub)] != 0)
message(sprintf("SLR hub edges: smoker=%d, non-smoker=%d", n_slr_sm_hub, n_slr_ns_hub))

net_slr_hub   <- netConstruct(data=pcor_slr_sm_hub, data2=pcor_slr_ns_hub,
                               dataType="condDependence", sparsMethod="none",
                               normMethod="none", verbose=0, seed=123456)
props_slr_hub <- netAnalyze(net_slr_hub, clustMethod="cluster_fast_greedy", verbose=FALSE)

message("Saving KORA SLR hub combined plot ...")
for (ext in c("png","svg")) {
  out <- file.path(FIG_DIR, paste0("netcomi_slr_kora_combined.", ext))
  if (ext=="png") png(out, width=4800, height=2400, res=300)
  else            svg(out, width=16, height=8)
  plot(props_slr_hub,
       groupNames = c(paste0("SLR: Smoker  |  top ", N_HUBS, " hubs  |  ", n_slr_sm_hub, " edges"),
                      paste0("SLR: Non-Smoker  |  top ", N_HUBS, " hubs  |  ", n_slr_ns_hub, " edges")),
       sameLayout = TRUE,
       rmSingles  = FALSE,
       nodeColor  = "colorVec",
       colorVec   = node_cols_hub,
       featVecCol = phyla_hub,
       repulsion  = 1.2,
       labelScale = FALSE,
       cexLabels  = 0.75)
  dev.off()
  message(sprintf("  Saved: netcomi_slr_kora_combined.%s", ext))
}

message("KORA genus sparse NetCoMi plots done.")
