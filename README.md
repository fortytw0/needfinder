# needfinder
Cheap and Fast - But is it better?

Setting this code up at the moment is quite straightforward : 

### Setup Conda Environment

`conda create --prefix=./venv/ --file conda.env` 
`source activate ./venv/`

### Main experiment runner 

`python -m src.main -config config/demo.json` will read from config.demo json and write to data/results in a directory timestamped with the unix time

### Run the Arora Similarity  

`python -m src.similarities.arora_similarity`

### Run the Domain Similarity

`python -m src.experiments.counts`
`python -m src.experiments.embeddings`
`python -m src.experiments.domain_similarity`

### Run SBERT similarity 

`python -m src.similarities.sbert_similarity`

You *will not* be able to run the scripts on multiple files for the timebeing, because we are still testing how the program should be structured to accomodate multiple subreddits. This will however, get you started. 
