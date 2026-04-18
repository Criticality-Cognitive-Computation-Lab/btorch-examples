#!/bin/bash
#SBATCH --job-name=glif_12ax
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/12ax_%A_%a.out

set -euo pipefail

SEEDS=(7 13 23 37 53)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
PROFILE=${PROFILE:-paper}

mkdir -p logs outputs/12ax

micromamba run -n ml-py312 python -m glif_net.experiments.task_12ax \
  seed=${SEED} \
  profile=${PROFILE} \
  output_dir=./outputs/12ax
