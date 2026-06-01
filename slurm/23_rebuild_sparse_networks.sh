#!/bin/bash
#SBATCH --job-name=sparse_netcomi
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=slurm/logs/sparse_netcomi_%j.out
#SBATCH --error=slurm/logs/sparse_netcomi_%j.err

set -euo pipefail

cd /home/itg/oleg.vlasovets/slr_example/Causal_Sparse_Low_Rank_Microbiome_Tutorial

source ~/miniconda3/etc/profile.d/conda.sh
conda activate slr-env

echo "=== $(date) === AGP: sparse NetCoMi (correct py_sparse_theta, 24/41 edges) ==="
Rscript analysis/plot_sparse_netcomi_agp.R

echo "=== $(date) === KORA genus: sparse NetCoMi (repulsion=1.8, ~310/304 edges) ==="
Rscript analysis/plot_sparse_netcomi_kora_genus.R

echo "=== $(date) === Done ==="
echo "AGP:"
find /home/itg/oleg.vlasovets/slr_example/Causal_Sparse_Low_Rank_Microbiome_Tutorial/results/two_group/figures -name "netcomi_sparse*.png" | sort
echo "KORA genus:"
find /home/itg/oleg.vlasovets/slr_example/KORA_Smoking_SLR/results/genus/figures -name "netcomi_sparse*.png" | sort
