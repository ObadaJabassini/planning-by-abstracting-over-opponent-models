#!/bin/bash
#SBATCH --job-name=training_static
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out
#SBATCH --time=24:00:00

module load anaconda3
python -u learn.py --nb-processes 24 --seed 32 --nb-steps 20 --save-interval 60 --opponent-class simple --with-monitoring