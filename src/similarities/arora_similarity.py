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
class AroraBeam(object) : 

    def __init__(self, alpha:float=1e-4) -> None:
        
        self.alpha = alpha
        self.word_embeddings = {}
        self.word_counts = {}
        self.word_probabilities = {}
        self.vocab = []
        self.num_total_words = None
        self.embedding_dimension = None

    def fit(self, jsonl_fpath:str, 
            embedding_fpath:str, 
            content_field_name:str='content') -> None : 
        """
        1. Given a jsonl file that contains all the Reddit posts, get all the content from the 
        jsonl file and extract all the post contents. 
        2. Given an embedding file, create a dictionary to map each word to its respective embedding.

        Args:
            jsonl_fpath (str): Path to the Reddit posts, captured in a jsonl file.
            
            embedding_fpath (str): Path to the embedding file. We assume that the word and 
            its respective embedding is delimited by tab. 
            
            content_field_name (str, optional): The field name in the Reddit jsonl file that contains 
            the content. Defaults to 'content'.
        """
        self._build_embeddings(embedding_fpath)
        self._count_words_in_reddit_posts(jsonl_fpath, content_field_name)
        





    def _get_sentence_embedding(self, sentence:str) -> None : 

        '''
        sum = 0
        for word in sentence : 
            smoothing_factor = alpha/(alpha + word_probabilities[word])
            word_vector = word_embeddings[word]
            sum +=  smoothing_factor*word_vector
        
        sentence_embedding = sum/len(sentence)

        '''

        sum = 0 

        for word in sentence : 
            smoothing_factor = self.alpha / (self.alpha + self.word_probabilities[word])
            word_vector = self.word_embeddings[word]
            sum+=smoothing_factor*word_vector

        sentence_embedding = sum/len(sentence)
        return sentence_embedding


    def _get_context_embedding(self, context:str) -> None : 

        '''

            context_matrix = np.zeros( len(word_vector) , len(context))

            for i, sentence in enumerate(context) : 
                context_matrix[: , i] = self._get_sentence_embedding(sentence)

            U, S, VT = svds(context_matrix, k=1)
            
            for i, sentence in enumerate(context) : 
                context_matrix[: , i] -= U@U.T@context_matrix[: , i]

            return context_matrix

        '''

        context_matrix = np.zeros((self.embedding_dimension , context))
        
        for i, sentence in enumerate(context) : 
            context_matrix[: , i] = self._get_sentence_embedding(sentence)

        U, S, VT = svds(context_matrix, k=1)

        for i, sentence in enumerate(context) : 
                context_matrix[: , i] -= U@U.T@context_matrix[: , i]

        return context_matrix


    def _count_words_in_reddit_posts(self, 
                                    jsonl_fpath:str, 
                                    content_field_name:str='content') : 

        """Given a jsonl file contain Reddit comments, it counts the number of words in each sample
        and updates self.num_total_words, self.word_counts, self.word_probabilities

        Args:
            jsonl_fpath (str): Path to Reddit comment submission. 

            content_field_name (str, optional): The field in the json that contains the 
            content of the post. Defaults to 'content'.
        """


        contents = []
        vectorizer = CountVectorizer()
        
        with open(jsonl_fpath) as f : 

            for i, line in enumerate(f.readlines()) : 
                reddit_post = json.loads(line)

                if content_field_name in reddit_post : 
                    contents.append(reddit_post[content_field_name])

        word_count_matrix = vectorizer.fit_transform(contents)
        word_counts = word_count_matrix.toarray().sum(axis=0)
        words = vectorizer.get_feature_names()
        self.vocab = words
        self.num_total_words = np.sum(word_counts)

        assert len(word_counts) == len(words)

        for w, wc in zip(words, word_counts) : 
            self.word_counts[w] = wc
            self.word_probabilities[w] = wc/self.num_total_words

    def _build_embeddings(self, embedding_fpath:str)->None : 

        with open(embedding_fpath) as f :
            for line in f.readlines() :
                word, embedding = line.split(' ', 1)

                embedding = np.fromstring(embedding, sep=' ')

                if word in self.vocab : 
                    self.word_embeddings[word] = embedding



if __name__ == '__main__' : 

    arora_beam = AroraBeam()
    arora_beam.fit(jsonl_fpath='data/airbnb_hosts.jsonl',
    embedding_fpath='data/glove.6B.300d.txt',
    content_field_name='body')
    print(arora_beam.word_counts)
    

