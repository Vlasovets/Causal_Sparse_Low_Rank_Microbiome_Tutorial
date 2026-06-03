#!/bin/bash
#SBATCH --job-name=regen_svg
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=slurm/logs/regen_svg_%j.out
#SBATCH --error=slurm/logs/regen_svg_%j.err

set -euo pipefail
cd /home/itg/oleg.vlasovets/slr_example/Causal_Sparse_Low_Rank_Microbiome_Tutorial

source ~/miniconda3/etc/profile.d/conda.sh
conda activate slr-env

# ── 1. AGP PERMANOVA null distribution (R — saves PNG + SVG) ─────────────────
echo "=== $(date) === PERMANOVA null distribution (AGP Aitchison) ==="
Rscript analysis/run_permanova.R

# ── 2. KORA: alpha-div + DA + classification (Python — saves PNG + SVG) ───────
echo "=== $(date) === KORA alpha-diversity, DA, classification ==="
python analysis/regenerate_svg_plots.py

echo "=== $(date) === Done ==="
echo "SVG outputs:"
find results/kora/svg_exports -name "*.svg" | sort
find results/two_group/figures -name "permanova_null_euclidean.svg"
