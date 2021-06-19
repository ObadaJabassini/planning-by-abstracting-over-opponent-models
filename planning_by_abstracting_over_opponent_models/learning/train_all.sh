#!/bin/sh
sbatch train_static_agent_slurm.sh
sbatch train_simple_agent_slurm.sh
sbatch train_random_agent_slurm.sh
sbatch train_mixed_agent_slurm.sh