#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=amilan
#SBATCH --time=12:00:00
#SBATCH --ntasks=24
#SBATCH --job-name=experiment2-job
#SBATCH --output=experiment2-job.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abha4861@colorado.edu

# module purge
# module load cuda
# module load cudnn

# nvidia-smi

source /curc/sw/anaconda3/latest
conda activate need

mkdir -p data/experiment2

python -m src.experiments.experiment2_autoeval
