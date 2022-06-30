from src.corpus import Corpus 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class SBERTSim(object) : 

    def __init__(self, 
                corpus:Corpus,  
                community:str, 
                model_name:str='paraphrase-MiniLM-L3-v2', 
                eager_load = False) -> None:
        
        self.corpus = corpus
        self.community = community
        self.model = SentenceTransformer(model_name)

        if eager_load : 
            self.matrix = None
        else : 
            self.matrix = self._fit_corpus()

    def _fit(self, sentences) : 
        return self.model.encode(sentences)

    def _fit_corpus(self) : 
        print('Finding contextual embeddings of sentences...')
        return self._fit(self.corpus.data)


    def _similarity(self, sentences) : 
        
        if self.matrix == None : 
            self.matrix = self._fit_corpus()

        sentence_repr = self._fit(sentences)
        return cosine_similarity(self.matrix, sentence_repr)

    def rank(self, sentences, save_path='sbert_results.csv') : 
        sim = self._similarity(sentences)
        return sim



        

if __name__ == '__main__' : 

    corpus = Corpus(['data/airbnb_hosts.jsonl'])
    
    sbert_sim = SBERTSim(corpus, 'airbnb_hosts')

    print(sbert_sim.matrix.shape)

    import json
    with open('data/labels.json') as f: 
        quotes = json.load(f)[1]['quotes']

    sbert_sim.rank(quotes)
