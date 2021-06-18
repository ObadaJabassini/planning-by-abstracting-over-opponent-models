#!/bin/bash
#SBATCH --job-name=training_static
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --error=slurm-static.err
#SBATCH --output=slurm-static.out
#SBATCH --time=60:05:00

export PYTHONPATH="$HOME/obada/planning-by-abstracting-over-opponent-models"
cd "$HOME/obada/planning-by-abstracting-over-opponent-models/planning_by_abstracting_over_opponent_models/learning"
python -u learn.py --nb-processes 24 --save-interval 60 --opponent-classes "static, static, static"