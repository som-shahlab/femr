#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --array=0-10
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu,nigam-a100
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=secure-gpu-3

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

python text_featurizer_temp.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
