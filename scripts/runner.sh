#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --output=sample-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dasr8731@colorado.edu


cd /scratch/summit/dasr8731/needfinder


python -m src.embedding_similarity
