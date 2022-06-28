import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


from src.similarities.arora_similarity import AroraBeam
from src.similarities.sbert_similarity import SBERTSim

from src.corpus import Corpus
from src.wordvectors.word2vec import W2VEmbedding
from src.wordvectors.glove import GloveEmbedding

'''
User Defined Variables.
'''

corpus_files = ['data/airbnb_hosts.jsonl']
embedding_type = 'word2vec' # 'word2vec' or 'glove'
community_name = 'airbnb_hosts' # what should the community embeddings be called? 
embedding_dimension = 100


'''
Read Quotes from CHI papers
'''

import json
with open('data/labels.json') as f: 
    quotes = json.load(f)

calorie_count_quotes = quotes[0]['quotes']
airbnb_quotes = quotes[1]['quotes']
gaming_quotes = quotes[2]['quotes']


'''
Program Variables
'''

corpus = Corpus(corpus_files)


if embedding_type == 'word2vec' : 

    # set retrain=True if you want to retrain word2vec embeddings.
    # it will automatically train if it doesn't find embeddings. 

    embedding = W2VEmbedding(corpus, 
                            savedir='data/wordvectors', 
                            community='airbnb_hosts', 
                            dimension=embedding_dimension, 
                            retrain=False)
else : 
    embedding =  GloveEmbedding(save_dir='data/glove', 
                            dimension=100)



asim = AroraBeam(embedding=embedding, 
                corpus=corpus, 
                embedding_dimension=100)

sbert = SBERTSim(corpus=Corpus, 
                community='airbnb_hosts' , 
                model_name='paraphrase-MiniLM-L3-v2')

'''
Sample usages
'''

# Standard Arora similarity. 
# Feed in the CHI quotes to compare with the respective corpus. 
arora_similarity = asim.rank(airbnb_quotes)
df = pd.DataFrame(arora_similarity, index=corpus.data , columns=airbnb_quotes)
df.to_csv('data/results/arora_sim.csv')


# Wordwise Arora similarity. 
# Feed in 1 CHI quote and 1 Reddit Post
post_labels, post_vectors, quote_labels, quote_vectors = asim.word_wise_rank(quote='CHI quote goes here.', 
                                                                            post='Reddit Post goes here.')
sim = cosine_similarity(post_vectors, quote_vectors)
df = pd.DataFrame(sim, index=post_labels, columns=quote_labels)
df.to_csv('data/results/arora_sim_wordwise_june21.csv')


# SBert Similarity
# Feed in the CHI quotes to compare with the respective corpus.
sbert_similarity = sbert.rank(airbnb_quotes)
df = pd.DataFrame(sbert_similarity, index=corpus.data , columns=airbnb_quotes)
df.to_csv('data/results/sbert_sim.csv')


