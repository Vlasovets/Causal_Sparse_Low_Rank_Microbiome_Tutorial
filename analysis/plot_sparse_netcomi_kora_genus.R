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

# ── Step 1: Build sparse hub network and extract reference layout ─────────────
N_HUBS <- 30
message(sprintf("Computing top %d hub nodes by union degree ...", N_HUBS))
adj_union    <- (abs(pcor_sm) > 0) | (abs(pcor_ns) > 0); diag(adj_union) <- FALSE
union_degree <- rowSums(adj_union)
top_nodes    <- names(sort(union_degree, decreasing=TRUE))[1:N_HUBS]
message("Top hub genera: ", paste(head(top_nodes, 6), collapse=", "), " ...")

pcor_sm_hub  <- pcor_sm[top_nodes, top_nodes]
pcor_ns_hub  <- pcor_ns[top_nodes, top_nodes]
node_cols_hub <- node_cols[top_nodes]
phyla_hub     <- phyla[top_nodes]

net_hub_sp   <- netConstruct(data=pcor_sm_hub, data2=pcor_ns_hub,
                              dataType="condDependence", sparsMethod="none",
                              normMethod="none", verbose=0, seed=123456)
props_hub_sp <- netAnalyze(net_hub_sp, clustMethod="cluster_fast_greedy", verbose=FALSE)

# Reference layout from sparse hub with rmSingles="inboth"
p_ref <- plot(props_hub_sp,
              groupNames = c("Smoker", "Non-Smoker"),
              sameLayout = TRUE,
              rmSingles  = "inboth",
              nodeColor  = "colorVec",
              colorVec   = node_cols_hub,
              featVecCol = phyla_hub,
              repulsion  = 1.2,
              labelScale = FALSE,
              cexLabels  = 0.75)

reference_layout <- p_ref$layout$layout1
reference_taxa   <- rownames(reference_layout)
message(sprintf("Reference hub taxa (with >=1 edge in sparse): %d of %d",
                length(reference_taxa), N_HUBS))

# ── Load permApprox significant edges ─────────────────────────────────────────
perm_file <- file.path(SPARSE, "sparse_perm_edge_pvals.csv")
sig_pairs <- data.frame(taxon_i=character(), taxon_j=character(), stringsAsFactors=FALSE)
if (file.exists(perm_file)) {
  perm_df  <- read.csv(perm_file, stringsAsFactors=FALSE)
  sig_rows <- perm_df[perm_df$bh_pval < 0.1, c("taxon_i","taxon_j")]
  sig_pairs <- sig_rows
  message(sprintf("permApprox significant edges (FDR<0.1): %d", nrow(sig_pairs)))
}

# ── Step 2: Sparse combined plot (reference_taxa + reference_layout) ───────────
message("Saving KORA sparse hub combined plot ...")
pcor_sm_ref <- pcor_sm_hub[reference_taxa, reference_taxa]
pcor_ns_ref <- pcor_ns_hub[reference_taxa, reference_taxa]
n_sm_hub <- sum(pcor_sm_ref[upper.tri(pcor_sm_ref)] != 0)
n_ns_hub <- sum(pcor_ns_ref[upper.tri(pcor_ns_ref)] != 0)
message(sprintf("Sparse hub edges: smoker=%d, non-smoker=%d", n_sm_hub, n_ns_hub))

net_sp_ref   <- netConstruct(data=pcor_sm_ref, data2=pcor_ns_ref,
                              dataType="condDependence", sparsMethod="none",
                              normMethod="none", verbose=0, seed=123456)
props_sp_ref <- netAnalyze(net_sp_ref, clustMethod="cluster_fast_greedy", verbose=FALSE)

for (ext in c("png","svg")) {
  out <- file.path(FIG_DIR, paste0("netcomi_sparse_kora_combined.", ext))
  if (ext=="png") png(out, width=4800, height=2400, res=300)
  else            svg(out, width=16, height=8)
  plot(props_sp_ref,
       groupNames = c(paste0("Smoker  |  top ", N_HUBS, " hubs  |  ", n_sm_hub, " edges"),
                      paste0("Non-Smoker  |  top ", N_HUBS, " hubs  |  ", n_ns_hub, " edges")),
       layout     = reference_layout,
       sameLayout = TRUE,
       rmSingles  = FALSE,
       nodeColor  = "colorVec",
       colorVec   = node_cols_hub[reference_taxa],
       featVecCol = phyla_hub[reference_taxa],
       repulsion  = 1.2,
       labelScale = FALSE,
       cexLabels  = 0.75)
  dev.off()
  message(sprintf("  Saved: netcomi_sparse_kora_combined.%s", ext))
}

# ── Step 3: SLR combined plot (SAME reference_taxa + reference_layout) ─────────
message("\n=== KORA SLR network (same superset + layout as sparse) ===")
theta_slr_sm <- as.matrix(read.csv(file.path(KORA_ROOT,"results/genus/slr/py_slr_theta_smoker.csv"), row.names=1, check.names=FALSE))
theta_slr_ns <- as.matrix(read.csv(file.path(KORA_ROOT,"results/genus/slr/py_slr_theta_non_smoker.csv"), row.names=1, check.names=FALSE))
pcor_slr_sm  <- theta_to_pcor(theta_slr_sm)
pcor_slr_ns  <- theta_to_pcor(theta_slr_ns)

# Subset to hub superset, then to reference_taxa
pcor_slr_sm_ref <- pcor_slr_sm[reference_taxa, reference_taxa]
pcor_slr_ns_ref <- pcor_slr_ns[reference_taxa, reference_taxa]
n_slr_sm_hub <- sum(pcor_slr_sm_ref[upper.tri(pcor_slr_sm_ref)] != 0)
n_slr_ns_hub <- sum(pcor_slr_ns_ref[upper.tri(pcor_slr_ns_ref)] != 0)
message(sprintf("SLR hub edges (reference taxa): smoker=%d, non-smoker=%d", n_slr_sm_hub, n_slr_ns_hub))

net_slr_ref   <- netConstruct(data=pcor_slr_sm_ref, data2=pcor_slr_ns_ref,
                               dataType="condDependence", sparsMethod="none",
                               normMethod="none", verbose=0, seed=123456)
props_slr_ref <- netAnalyze(net_slr_ref, clustMethod="cluster_fast_greedy", verbose=FALSE)

for (ext in c("png","svg")) {
  out <- file.path(FIG_DIR, paste0("netcomi_slr_kora_combined.", ext))
  if (ext=="png") png(out, width=4800, height=2400, res=300)
  else            svg(out, width=16, height=8)
  plot(props_slr_ref,
       groupNames = c(paste0("SLR: Smoker  |  top ", N_HUBS, " hubs  |  ", n_slr_sm_hub, " edges"),
                      paste0("SLR: Non-Smoker  |  top ", N_HUBS, " hubs  |  ", n_slr_ns_hub, " edges")),
       layout     = reference_layout,
       sameLayout = TRUE,
       rmSingles  = FALSE,
       nodeColor  = "colorVec",
       colorVec   = node_cols_hub[reference_taxa],
       featVecCol = phyla_hub[reference_taxa],
       repulsion  = 1.2,
       labelScale = FALSE,
       cexLabels  = 0.75)
  dev.off()
  message(sprintf("  Saved: netcomi_slr_kora_combined.%s", ext))
}

# ── Differential network ──────────────────────────────────────────────────────
message("\nComputing KORA sparse differential network ...")
diff_net_hub <- diffnet(net_sp_ref, diffMethod="fisher", n1=236, n2=236)
n_diff_hub   <- sum(diff_net_hub$diffMat != 0, na.rm=TRUE)
if (n_diff_hub > 0) {
  diff_edge_col <- c("#CC79A7","#009E73","#0072B2","#E69F00","#999999","#56B4E9","#F0E442","white","#B55E00")
  for (ext in c("png","svg")) {
    out <- file.path(FIG_DIR, paste0("netcomi_sparse_diff_kora_hub.", ext))
    if (ext=="png") png(out, width=3200, height=2600, res=300)
    else            svg(out, width=10.67, height=8.67)
    plot(diff_net_hub, rmSingles=FALSE, mar=c(2,2,5,10),
         edgeCol=diff_edge_col, edgeWidth=2.5, labelScale=FALSE, cexLabels=0.70)
    dev.off()
    message(sprintf("  Saved: netcomi_sparse_diff_kora_hub.%s", ext))
  }
}

message("KORA genus sparse + SLR NetCoMi plots done.")
