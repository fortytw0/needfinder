from gensim.models import KeyedVectors
from src.similarities.similarity import Sim
from src.wordvectors.word2vec import W2VEmbedding

from src.corpus import Corpus


import time
from tqdm import tqdm
import numpy as np
import re
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity


class AroraBeam(object) : 


    def __init__(self, embedding:W2VEmbedding, 
                corpus:Corpus,  
                community:str, 
                alpha:float=1e-4,  
                embedding_dimension:int=50) -> None:
        
        # User Defined
        self.alpha = alpha
        self.corpus = corpus
        self.community = community
        self.word_embeddings = embedding.embeddings[self.community]
        self.vocab = self.corpus.vocabs['unigram'][self.community]
        self.word_probabilities = self.corpus.term_frequency_matrix['unigram'][self.community].to_dict()
        self.embedding_dimension = embedding_dimension      
        self.unk_words = 0
        print("Preparing Corpus embedding repr...")
        self.corpus_repr = self._fit_corpus()

    
    def _fit_corpus(self) -> np.array : 
        return self._fit(self.corpus.corpus[self.community]) 


    def _fit(self, texts: list) -> np.array:
        
        text_repr = []
        for text in tqdm(texts) : 
            text_repr.append(self._get_sentence_embedding(text))

        return np.array(text_repr)

    def rank(self, queries) -> float:
        print('Preparing queries embedding repr... ')
        text_repr = self._fit(queries)
        return cosine_similarity(text_repr, self.corpus_repr)

    def _get_sentence_embedding(self, sentence:str) -> None : 

        sentence = self._process_sentence(sentence)
        if len(sentence) == 0 : 
            return np.zeros((self.embedding_dimension,))

        sum = 0 

        for word in sentence : 

            try : 
                if word in self.word_probabilities :     
                    smoothing_factor = self.alpha / (self.alpha + self.word_probabilities[word])
                    word_vector = self._get_word_embedding(word)
                    sum+=smoothing_factor*word_vector

                else : 
                    self.unk_words += 1

            except Exception as e :

                print(e)
                print(self._get_word_embedding(word))
                print('Word : ' , word)


        sentence_embedding = sum/len(sentence)
        return sentence_embedding


    def _get_context_embedding(self, context:str) -> None : 

        context_matrix = np.zeros((self.embedding_dimension , len(context)))
        
        for i, sentence in enumerate(context) : 
            context_matrix[: , i] = self._get_sentence_embedding(sentence)

        U, S, VT = svds(context_matrix, k=1)

        for i, sentence in enumerate(context) : 
                context_matrix[: , i] -= U@U.T@context_matrix[: , i]

        return context_matrix

    def _get_word_embedding(self, word:str) -> np.array : 

        if word in self.word_embeddings.key_to_index.keys() : 
            return self.word_embeddings[word]

        return self.word_embeddings['<UNK>']

    def _process_sentence(self, sentence) : 

        tokens = re.findall(r'(\w+)' , sentence)
        tokens = [t for t in tokens if len(t) > 1]
        tokens = [t if t in self.vocab else '<UNK>' for t in tokens]

        return tokens


if __name__ == '__main__' : 

    corpus = Corpus({'airbnb_hosts' : [{'subreddit' : 'airbnb_hosts' , 'subreddit_path' : 'data/airbnb_hosts.jsonl'}], 
                    'airbnb' : [{'subreddit' : 'airbnb' , 'subreddit_path' : 'data/airbnb.jsonl'}], 
                    'vrbo' : [{'subreddit' : 'vrbo' , 'subreddit_path' : 'data/vrbo.jsonl'}], 
                    'caloriecount' : [{'subreddit' : 'caloriecount' , 'subreddit_path' : 'data/caloriecount.jsonl'}],
                    'loseit' : [{'subreddit' : 'loseit' , 'subreddit_path' : 'data/loseit.jsonl'}],
                    })

    embedding = W2VEmbedding(corpus, 'data/wordvectors')

    community = 'airbnb_hosts'
    
    ab = AroraBeam(embedding, corpus, community, embedding_dimension=100)

    import json
    with open('data/labels.json') as f: 
        quotes = json.load(f)[1]['quotes']
        
    print(quotes)

    sim = ab.rank(quotes)

    print(sim)


    



    

    

