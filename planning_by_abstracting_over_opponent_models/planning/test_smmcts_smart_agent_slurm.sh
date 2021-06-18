#!/bin/bash
#SBATCH --job-name=smmcts_smart
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --error=slurm-smart.err
#SBATCH --output=slurm-smart.out
#SBATCH --time=48:05:00

export PYTHONPATH="$HOME/obada/planning-by-abstracting-over-opponent-models"
cd "$HOME/obada/planning-by-abstracting-over-opponent-models/planning_by_abstracting_over_opponent_models/planning"
python -u test_smmcts.py --model-iterations 2700 --opponent-classes "smart_no_bomb, smart_no_bomb, smart_no_bomb" --nb-games 10 --nb-plays 10 --mcts-iterations 5000 --fpu 1000