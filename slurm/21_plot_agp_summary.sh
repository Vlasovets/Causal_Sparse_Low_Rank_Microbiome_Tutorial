#!/bin/bash
#SBATCH --job-name=agp_plots
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_preemptible
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=slurm/logs/agp_plots_%j.out
#SBATCH --error=slurm/logs/agp_plots_%j.err

set -euo pipefail

cd /home/itg/oleg.vlasovets/slr_example/Causal_Sparse_Low_Rank_Microbiome_Tutorial

source ~/miniconda3/etc/profile.d/conda.sh
conda activate slr-env

echo "=== $(date) === AGP summary plots ==="
python analysis/plot_agp_summary.py

echo "=== $(date) === Done ==="
find plots -name "agp_*" -o -name "*venn*" | sort
