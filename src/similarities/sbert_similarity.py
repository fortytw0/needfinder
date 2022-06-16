from src.corpus import Corpus 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class SBERTSim(object) : 

    def __init__(self, 
                corpus:Corpus,  
                community:str, 
                model_name:str='paraphrase-MiniLM-L3-v2') -> None:
        
        self.corpus = corpus
        self.community = community
        self.model = SentenceTransformer(model_name)

        print('Finding contextual embeddings of sentences...')
        self.matrix = self._fit_corpus()

    def _fit(self, sentences) : 
        return self.model.encode(sentences)

    def _fit_corpus(self) : 
        return self._fit(self.corpus.corpus[self.community])


    def _similarity(self, sentences) : 
        sentence_repr = self._fit(sentences)
        return cosine_similarity(self.matrix, sentence_repr)

    def rank(self, sentences, save_path='sbert_results.csv') : 
        sim = self._similarity(sentences)
        df = pd.DataFrame(sim, index=self.corpus.corpus[self.community] , columns=sentences)
        df.to_csv(save_path)
        return df



        

if __name__ == '__main__' : 

    corpus = Corpus({'airbnb_hosts' : [{'subreddit' : 'airbnb_hosts' , 'subreddit_path' : 'data/airbnb_hosts.jsonl'}], 
                    'airbnb' : [{'subreddit' : 'airbnb' , 'subreddit_path' : 'data/airbnb.jsonl'}], 
                    'vrbo' : [{'subreddit' : 'vrbo' , 'subreddit_path' : 'data/vrbo.jsonl'}], 
                    'caloriecount' : [{'subreddit' : 'caloriecount' , 'subreddit_path' : 'data/caloriecount.jsonl'}],
                    'loseit' : [{'subreddit' : 'loseit' , 'subreddit_path' : 'data/loseit.jsonl'}],
                    })
    
    sbert_sim = SBERTSim(corpus, 'airbnb_hosts')

    print(sbert_sim.matrix.shape)

    import json
    with open('data/labels.json') as f: 
        quotes = json.load(f)[1]['quotes']

    sbert_sim.rank(quotes)
