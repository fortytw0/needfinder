#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=shas
#SBATCH --ntasks=24
#SBATCH --job-name=experiment2-job
#SBATCH --output=experiment2-job.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dasr8731@colorado.edu

# module purge
# module load cuda
# module load cudnn

# nvidia-smi

cd /scratch/summit/dasr8731/needfinder
source activate ./venv/ 

python -m src.experiments.experiment2_autoeval
