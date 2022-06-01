# needfinder
Cheap and Fast - But is it better?

Setting this code up at the moment is quite straightforward : 

1. Setup Conda Environment

`conda create --prefix=./venv/ --file conda.env` 
`source activate ./venv/`

2. Run the Arora Similarity  

`python -m src.similarities.arora_similarity`

3. Run the Domain Similarity

`python -m src.experiments.counts`
`python -m src.experiments.embeddings`
`python -m src.experiments.domain_similarity`

4. Run SBERT similarity 

`python -m src.similarities.sbert_similarity`

You *will not* be able to run the scripts on multiple files for the timebeing, because we are still testing how the program should be structured to accomodate multiple subreddits. This will however, get you started. 
