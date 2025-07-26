#!/bin/bash
#SBATCH --job-name=run_finetune
#SBATCH --output=logs/finetune.out
#SBATCH --error=logs/finetune.err
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00

echo "Starting at $(date) on $(hostname)"

# Load conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asr_env

# Safety flags
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_READ_TIMEOUT=60
export TOKENIZERS_PARALLELISM=false

# Launch the script
python scripts/fine_tune.py

echo "Finished at $(date)"

