#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=12:10:00
#SBATCH --output=sample-%j.out

module purge
module load anaconda

cd /projects/dasr8731/needfinder

conda activate needfinder0.0

python -m src.reddit_scraper
