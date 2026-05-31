#!/bin/bash
#SBATCH --job-name=agp_alpha_div
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/agp_alpha_div_%j.out
#SBATCH --error=slurm/logs/agp_alpha_div_%j.err

set -euo pipefail

cd /home/itg/oleg.vlasovets/slr_example/Causal_Sparse_Low_Rank_Microbiome_Tutorial

export TMPDIR=/localscratch/${USER}/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate slr-env

echo "=== $(date) === AGP: Breakaway richness + DivNet Shannon ==="
python analysis/run_alpha_diversity_agp.py

echo "=== $(date) === Done ==="
echo "Outputs:"
find results/agp/alpha_diversity -type f | sort
