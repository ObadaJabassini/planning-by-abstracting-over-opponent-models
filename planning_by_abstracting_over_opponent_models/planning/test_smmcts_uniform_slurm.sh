#!/bin/bash
#SBATCH --job-name=smmcts_uniform
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --error=slurm_smmcts_uniform_%A_%a.err
#SBATCH --output=slurm_smmcts_uniform_%A_%a.out
#SBATCH --time=12:05:00
#SBATCH --array=1-10

export PYTHONPATH="$HOME/obada/planning-by-abstracting-over-opponent-models"
cd "$HOME/obada/planning-by-abstracting-over-opponent-models/planning_by_abstracting_over_opponent_models/planning"
python -u test_smmcts.py --policy-estimation uniform --opponent-classes "simple, simple, simple" --nb-games 2 --nb-plays 10 --mcts-iterations 2000 --fpu 1000