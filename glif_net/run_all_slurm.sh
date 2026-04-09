#!/bin/bash
#SBATCH --job-name=glif_tutorials
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G

TASKS=("speech_command" "shd" "seqmnist" "poisson_mnist")
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "Job: $SLURM_JOB_ID, Task: $TASK, Host: $(hostname)"

mkdir -p logs checkpoints runs data

# Run with task-specific defaults (no YAML files needed)
micromamba run -n ml-py312 python -m glif_net.run_tutorial \
	task=$TASK \
	epochs=100 \
	calibrate=True \
	use_tensorboard=True \
	device=cuda

echo "Task $TASK completed"
