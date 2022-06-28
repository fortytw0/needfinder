#!/bin/bash

#SBATCH --partition=amilan-ucb
#SBATCH --job-name=run_needfinder
#SBATCH --output=run_needfinder.out
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# module load slurm/summit to access summit scheduler

# run $sbatch setup.slurm to make the sub corpora

source /curc/sw/anaconda3/latest
conda activate need
python -m src.main
