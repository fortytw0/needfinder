#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=12:00:00

#SBATCH --partition=sgpu
#SBATCH --job-name=experiment2-job
#SBATCH --output=experiment2-job.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dasr8731@colorado.edu

module purge
module load cuda
module load cudnn

nvidia-smi

export SENTENCE_TRANSFORMERS_HOME=/scratch/summit/dasr8731/models
export TRANSFORMERS_CACHE=/scratch/summit/dasr8731/models

cd /scratch/summit/dasr8731/needfinder
source activate ./gpuenv/ 

python -m src.experiment2.runner > experiment2.log

##SBATCH --ntasks=24