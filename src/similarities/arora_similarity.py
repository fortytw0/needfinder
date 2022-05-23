from torch import embedding
from src.similarities.similarity import Sim

import time
from tqdm import tqdm
import numpy as np
import re
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
        self.unk_words = 0
        
        # Will be declared/calculated
        self.vectorizer = None
        

    def build(self, jsonl_path: str, content_field: str, id_field:str) -> None:
        content, ids = super().build(jsonl_path, content_field, id_field)
        self.ids = ids

        self._build_vocab(content)
        self._build_embeddings()
        self._build_probabilities()

        self.matrix = np.zeros((self.embedding_dimension, ))

        for i in tqdm(range(len(content))) : 
             
            sentence = content[i]
            embedding = self._get_sentence_embedding(sentence) 

            try :
                self.matrix = np.vstack((self.matrix, embedding))

            except Exception as e :
                print(e)
                self.matrix = np.vstack((self.matrix, np.zeros((self.embedding_dimension, ))))

        self.matrix = self.matrix[1:, :]

    def fit(self, texts: list) -> np.array:
        
        text_repr = []

        for text in texts : 
            text_repr.append(self._get_sentence_embedding(text))

        return np.array(text_repr)

    def similarity(self, text_repr: np.array) -> float:
        return super().similarity(text_repr)

    
        
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


    def _build_vocab(self, content:list) : 

        self.vectorizer = CountVectorizer(token_pattern=r"(\w+)")

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

        # prepare unknown embedding
        self.word_embeddings['<UNK>'] = np.mean(list(self.word_embeddings.values()), axis=0)
        

    def _build_probabilities(self) -> None : 

        for w, wc in self.word_counts.items() : 
            self.word_probabilities[w] = wc/self.num_total_words

    def _get_word_embedding(self, word:str) -> np.array : 

        if word in self.word_embeddings : 
            return self.word_embeddings[word]

        return self.word_embeddings['<UNK>']

    def _process_sentence(self, sentence:str) -> list:

        sentence = sentence.replace("'", '').strip().lower()
        sentence_matches = re.findall(r"(\w+)", sentence)
        return sentence_matches


if __name__ == '__main__' : 

    arora_beam = AroraBeam()
    arora_beam.build('data/airbnb_hosts.jsonl','body', 'body')
    
    import json
    with open('data/labels.json') as f : 
        data = json.load(f)
        print('Finished loading data.')
        airbnb_data = data[1]['quotes']
        print('Airbnb Data : \n' , airbnb_data)
    


    airbnb_data_repr = arora_beam.fit(airbnb_data)
    print(airbnb_data_repr)

    print(arora_beam.similarity(airbnb_data_repr).shape)

    import pandas as pd 

    df = pd.DataFrame(arora_beam.similarity(airbnb_data_repr), index=arora_beam.ids, columns=airbnb_data)
    df.to_csv('arora_similarity.csv')

    



    

    

