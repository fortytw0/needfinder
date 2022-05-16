from src.similarities.similarity import Sim

from functools import reduce
import numpy as np
import pandas as pd
import json
import spacy
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer


# Setting up Logging
import logging
logging.basicConfig(filename='logs/main.log',
                    level=logging.INFO, 
                    format='%(asctime)s :: %(levelname)s :: %(module)s :: %(message)s')
logging.info('Logging has been setup.')


# Loading Spacy
nlp = spacy.load('en_core_web_sm')


# AroraBeam 
class AroraBeam(Sim) : 


    def __init__(self, alpha:float=1e-4, embedding_path:str='data/glove.6B.50d.txt', embedding_dimension:int=50) -> None:

        super().__init__()
        
        # User Defined
        self.alpha = alpha
        self.embedding_dimension = embedding_dimension      
        self.embedding_path = embedding_path

        # Will be built
        self.word_embeddings = {}
        self.word_counts = {}
        self.word_probabilities = {}
        self.vocab = []
        self.num_total_words = 0
        
        # Will be declared/calculated
        self.vectorizer = None
        

    def build(self, jsonl_path: str, content_field: str) -> None:
        content = super().build(jsonl_path, content_field)
        self._build_vocab(content)
        self._build_embeddings()
        self._build_probabilities()

        '''
        Unable to resolve 2 things : 

        1. How to find embeddings for unknown words?
        2. Is there a quick and easy way to tokenize? Spacy takes too long. 
            - Looked up - regex is very quick, and probably gives a crude way of tokenization. 
            Turns out, sklearn's text vectorizer uses a similar approach. 
            - NLTK tokenize is another one we can consider, but it also takes too long (for the 
            scale that we are working on)
        
        '''


        




    def _get_sentence_embedding(self, sentence:str) -> None : 

        sum = 0 

        for word in sentence : 
            smoothing_factor = self.alpha / (self.alpha + self.word_probabilities[word])
            word_vector = self.word_embeddings[word]
            sum+=smoothing_factor*word_vector

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


    def _build_vocab(self, content:list) : 

        self.vectorizer = CountVectorizer()

        word_count_matrix = self.vectorizer.fit_transform(content)
        word_counts = word_count_matrix.toarray().sum(axis=0)
        self.vocab = self.vectorizer.get_feature_names_out()

        assert len(word_counts) == len(self.vocab)

        for w, wc in zip(self.vocab, word_counts) : 
            self.word_counts[w] = wc
            self.num_total_words += wc

 

    def _build_embeddings(self)->None : 

        with open(self.embedding_path) as f :
            for line in f.readlines() :
                word, embedding = line.split(' ', 1)

                if word in self.vocab : 
                    embedding = np.fromstring(embedding, sep=' ')
                    self.word_embeddings[word] = embedding


    def _build_probabilities(self) -> None : 

        for w, wc in self.word_counts.items() : 
            self.word_probabilities[w] = wc/self.num_total_words






if __name__ == '__main__' : 

    arora_beam = AroraBeam()
    arora_beam.build('data/airbnb_hosts.jsonl','body')
    print(arora_beam.word_counts)
    print(arora_beam.num_total_words)
    

