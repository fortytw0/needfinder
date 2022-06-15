#!/bin/bash

#SBATCH --partition=amilan-ucb
#SBATCH --job-name=make_subs
#SBATCH --output=make_subs.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# module load slurm/summit to access summit scheduler

cat ../../dj/reddits.jsonl | rg '"airbnb_hosts"' -F > data/airbnb_hosts.jsonl

cat ../../dj/reddits.jsonl | rg '"airbnb"' -F > data/airbnb.jsonl

cat ../../dj/reddits.jsonl | rg '"vrbo"' -F > data/vrbo.jsonl

cat ../../dj/reddits.jsonl | rg '"caloriecount"' -F > data/caloriecount.jsonl

cat ../../dj/reddits.jsonl | rg '"loseit"' -F > data/loseit.jsonl

