import json
import pandas as pd
import numpy as np


from src.similarities.arora_similarity import AroraBeam
from src.corpus import Corpus
from src.wordvectors.word2vec import W2VEmbedding



'''

README FOR PROGRAM VARIABLES

---xxx---

corpus_dictionary (dict) : Dictionary containing all the subreddits and communities.  

>>> Template : 
{community_name : [{subreddit : 'subreddit1' , subreddit_path : subreddit1.jsonl} , {subreddit : 'subreddit2' , subreddit_path : subreddit2.jsonl} ]}

In this example, an online community is assigned as a collection of different subreddits. 
The subreddit_path is the jsonl where the reddit posts are stored. 

---xxx---

vectorizer_definitions (dict) : How to build count vectorizers for the corpus. 

>>> Template :

[{max_features: max_vocab_size , vectorizer_type : name_of_ngram, ngram_range: tuple_of_ngram}]

ngram_range of (2 ,2) means a 2-gram. Please note, you always need to provide ngram as (a , b) where a==b. If a!=b, there will be bugs and low accuracy. 

---xxx---

quotes (list) : List of CHI quotes that user wants to match with reddit. 

---xxx---

wordvector_path (str) : Path to save wordvectors

---xxx---

community (str) : Community out of the given community names in corpus to focus/compare CHI quotes to.

---xxx---

output_path (str) : Where to save the ranked results. 

'''


corpus_dictionary = {'airbnb_hosts' : [{'subreddit' : 'airbnb_hosts' , 'subreddit_path' : 'data/airbnb_hosts.jsonl'}], 
                    'airbnb' : [{'subreddit' : 'airbnb' , 'subreddit_path' : 'data/airbnb.jsonl'}], 
                    'vrbo' : [{'subreddit' : 'vrbo' , 'subreddit_path' : 'data/vrbo.jsonl'}], 
                    'caloriecount' : [{'subreddit' : 'caloriecount' , 'subreddit_path' : 'data/caloriecount.jsonl'}],
                    'loseit' : [{'subreddit' : 'loseit' , 'subreddit_path' : 'data/loseit.jsonl'}],
                    }

vectorizer_definitions = [
                        {'max_features' : 1e4, 'vectorizer_type' : 'unigram' , 'ngram_range' : (1 , 1)},
                        ]



config = {"wordvector_path": "data/wordvectors",
         "community": "airbnb_hosts",
         "input_quotes": "data/labels.json",
         "output_path": "data/results/arora_similarity_results.csv",
         "corpus_dictionary": corpus_dictionary, 
         "community": 'airbnb_hosts',
         "embedding_dimension": 100,
         "vectorizer_definitions": vectorizer_definitions}

# Defining CHI quotes
with open(config['input_quotes']) as f: 
    quotes = json.load(f)[1]['quotes']

if __name__ == '__main__' :            

    corpus = Corpus(config["corpus_dictionary"], 
                    vectorizer_definitions=config["vectorizer_definitions"])
    embedding = W2VEmbedding(corpus, config["wordvector_path"])
    ab = AroraBeam(embedding, corpus, config["community"], 
                  config["embedding_dimension"])
    sim = ab.rank(quotes)

    import pandas as pd 

    df = pd.DataFrame(sim, index=corpus.corpus[config['community']], columns=quotes)
    df.to_csv(config['output_path'])

