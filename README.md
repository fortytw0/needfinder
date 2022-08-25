# needfinder
Cheap and Fast - But is it better?

Setting this code up at the moment is quite straightforward : 

### How to setup the conda Environment

`conda create --prefix=./venv/ --file conda.env` 
`source activate ./venv/`

### How to add a new paper

1. Add posts from the relevant online community in `data/` following the format in `data/demo.airbnb_hosts.jsonl`. The easiest thing to just have an id and body field, though the code is flexible and allows other keys.
2. Add labels to `data/labels.json` following the format in the file; note the paper title is the key in the JSON structure. This needs to match your config (see next step)
3. Add a config file to `config/[your config name].json` following the format in `config/demo.json`
4. To run your job using slurm on alpine do `sbatch --export=config="config/allnurses.json" scripts/run.alpine.slurm` where the export flag fills the environment variable config w/ your environment variable string

### Main experiment runner

`python -m src.main -config config/demo.json` will read from config.demo json and write to data/results in a directory timestamped with the unix time

- `scripts/experiment2_runner.sh` will run experiment 2

### Run the Arora Similarity

`python -m src.similarities.arora_similarity`

### Run the Domain Similarity

`python -m src.experiments.counts`
`python -m src.experiments.embeddings`
`python -m src.experiments.domain_similarity`

### Run SBERT similarity

`python -m src.similarities.sbert_similarity`

You *will not* be able to run the scripts on multiple files for the timebeing, because we are still testing how the program should be structured to accomodate multiple subreddits. This will however, get you started. 
