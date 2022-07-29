import os
import json
import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


from src.similarities.arora_similarity import AroraBeam
from src.similarities.sbert_similarity import SBERTSim

from src.corpus import Corpus
from src.wordvectors.word2vec import W2VEmbedding
from src.wordvectors.glove import GloveEmbedding

'''
User Defined Variables.
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '-config', '--config', action="store", dest="config", default="config/demo.json")
args = parser.parse_args()
print(args)

with open(args.config, "r") as inf:
    config = json.load(inf)

'''
Read Quotes from papers
'''

with open(config["interview_quotes"]) as f:
    title = config["papertitle"]
    paper_quotes = json.load(f)[title]["quotes"]

'''
Program Variables
'''

corpus = Corpus(config['corpus_files'])

if config['embedding_type'] == 'word2vec':

    # set retrain=True if you want to retrain word2vec embeddings.
    # it will automatically train if it doesn't find embeddings.

    embedding = W2VEmbedding(corpus,
                             savedir='data/wordvectors',
                             community=config["community_name"],
                             dimension=config['embedding_dimension'],
                             retrain=False)
elif config['embedding_type'] == 'glove':
    embedding = GloveEmbedding(save_dir='data/glove',
                               dimension=config['embedding_dimension'])

else:
    assert "There is a" == "problem"

sbert = SBERTSim(corpus=corpus,
                 community=config["community_name"],
                 model_name=config["model_name"])

asim = AroraBeam(embedding=embedding,
                 corpus=corpus,
                 eager_load=True,
                 embedding_dimension=config['embedding_dimension'])


'''
Sample usages
'''

output_folder = "data/results/{}/".format(int(time.time()))
os.mkdir(output_folder)

with open(output_folder + "config.json", "w") as of:
    json.dump(config, of)

# Standard Arora similarity.
# Feed in the CHI quotes to compare with the respective corpus.
arora_similarity = asim.rank(paper_quotes)
df = pd.DataFrame(arora_similarity, index=corpus.data, columns=paper_quotes)
df.to_csv(output_folder + 'arora_sim.csv')

# Wordwise Arora similarity.
# Feed in 1 CHI quote and 1 Reddit Post
post_labels, post_vectors, quote_labels, quote_vectors = asim.word_wise_rank(quote='CHI quote goes here.',
                                                                             post='Reddit Post goes here.')
sim = cosine_similarity(post_vectors, quote_vectors)
df = pd.DataFrame(sim, index=post_labels, columns=quote_labels)
df.to_csv(output_folder + 'arora_sim_wordwise_june21.csv')


# SBert Similarity
# Feed in the CHI quotes to compare with the respective corpus.
sbert_similarity = sbert.rank(paper_quotes)
df = pd.DataFrame(sbert_similarity, index=corpus.data, columns=paper_quotes)
df.to_csv(output_folder + 'sbert_sim.csv')
