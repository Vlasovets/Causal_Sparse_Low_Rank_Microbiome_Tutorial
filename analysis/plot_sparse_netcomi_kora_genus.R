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
tax        <- read.csv(TAX_F, row.names = 1, stringsAsFactors = FALSE)

get_phylum <- function(name) {
  if (name %in% rownames(tax)) {
    ph <- tax[name, "Phylum"]
    if (!is.na(ph) && nchar(ph) > 1) return(ph)
  }
  "Unknown"
}
phyla <- setNames(sapply(taxa_names, get_phylum), taxa_names)

pal <- c("#88CCEE","#CC6677","#DDCC77","#117733","#332288",
         "#AA4499","#44AA99","#999933","#882255","#661100",
         "#6699CC","#888888","#E69F00","#D55E00","#0072B2")
unique_phyla <- sort(unique(phyla))
phy_cols     <- setNames(pal[seq_along(unique_phyla)], unique_phyla)
node_cols    <- phy_cols[phyla]
names(node_cols) <- taxa_names

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
layout_ref <- p_ref$layout$layout1

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

message("KORA genus sparse NetCoMi plots done.")
