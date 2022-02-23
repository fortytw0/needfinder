#! /bin/bash

cd /projects/dasr8731/needfinder

conda activate reddit-needfinding-0.0

conda install -y --file requirements.txt

python -m src.reddit_scraper