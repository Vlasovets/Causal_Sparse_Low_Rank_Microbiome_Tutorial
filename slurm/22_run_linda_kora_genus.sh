#!/bin/bash
#SBATCH --job-name=kora_linda_genus
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_preemptible
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/kora_linda_genus_%j.out
#SBATCH --error=slurm/logs/kora_linda_genus_%j.err

set -euo pipefail

cd /home/itg/oleg.vlasovets/slr_example/Causal_Sparse_Low_Rank_Microbiome_Tutorial

export TMPDIR=/localscratch/${USER}/${SLURM_JOB_ID}
mkdir -p "${TMPDIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate slr-env

echo "=== $(date) === KORA: genus-level LinDA + Lee et al. (1000 perms) ==="
python analysis/run_linda_kora_genus.py

echo "=== $(date) === Done ==="
find results/kora/da_genus -type f | sort
