#!/bin/bash
#SBATCH --job-name=smmcts_static
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --error=slurm-smmcts-static.err
#SBATCH --output=slurm-smmcts-static.out
#SBATCH --time=48:05:00

export PYTHONPATH="$HOME/obada/planning-by-abstracting-over-opponent-models"
cd "$HOME/obada/planning-by-abstracting-over-opponent-models/planning_by_abstracting_over_opponent_models/planning"
python -u test_smmcts.py --policy-estimation neural_network --opponent-classes "static, static, static" --nb-games 10 --nb-plays 10 --mcts-iterations 2000 --fpu 1000