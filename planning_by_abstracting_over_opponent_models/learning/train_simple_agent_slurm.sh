#!/bin/bash
#SBATCH --job-name=training_simple
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --error=slurm-simple.err
#SBATCH --output=slurm-simple.out
#SBATCH --time=25:05:00

export PYTHONPATH="$HOME/obada/planning-by-abstracting-over-opponent-models"
cd "$HOME/obada/planning-by-abstracting-over-opponent-models/planning_by_abstracting_over_opponent_models/learning"
python -u learn.py --nb-processes 24 --seed 32 --nb-steps 20 --save-interval 60 --opponent-class simple --with-monitoring