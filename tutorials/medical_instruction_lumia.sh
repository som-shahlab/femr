#!/bin/bash
#SBATCH --job-name=lumia
#SBATCH --output=logs/job_%A.out
#SBATCH --error=logs/job_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal,gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
####SBATCH --exclude=secure-gpu-10


python medical_instruction_lumia.py
