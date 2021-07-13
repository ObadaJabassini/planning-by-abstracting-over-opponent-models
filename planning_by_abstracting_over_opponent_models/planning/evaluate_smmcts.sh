#!/bin/bash
#SBATCH --job-name=smmcts_evaluate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --error=slurm_evaluate_smmcts_%A_%a.err
#SBATCH --output=slurm_evaluate_smmcts_%A_%a.out
#SBATCH --time=12:05:00
#SBATCH --array=1-10

export PYTHONPATH="$HOME/obada/planning-by-abstracting-over-opponent-models"
cd "$HOME/obada/planning-by-abstracting-over-opponent-models/planning_by_abstracting_over_opponent_models/planning"
python -u evaluate_smmcts.py --exploration-coef 0.8 --fpu 0.5  --policy-estimation uniform --opponent-classes "simple, simple, simple" --nb-games 50 --nb-plays 1 --mcts-iterations 500