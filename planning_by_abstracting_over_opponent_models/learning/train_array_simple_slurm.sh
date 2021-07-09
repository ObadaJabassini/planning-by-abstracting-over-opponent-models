#!/bin/bash
#SBATCH --job-name=training_si
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --error=slurm_training_simple_%A_%a.err
#SBATCH --output=slurm_training_simple_%A_%a.out
#SBATCH --time=48:05:00
#SBATCH --array=1-5

export PYTHONPATH="$HOME/obada/planning-by-abstracting-over-opponent-models"
cd "$HOME/obada/planning-by-abstracting-over-opponent-models/planning_by_abstracting_over_opponent_models/learning"
python -u learn.py --opponent-classes "simple, simple, simple"