from gensim.models import KeyedVectors
from src.similarities.similarity import Sim
import re
import pandas as pd
import numpy as np


class DomainSim(Sim) : 

    def __init__(self, wv_save_path, term_prob_csv, community, 
                alpha=1e-3, unk_token='<UNK>') -> None:
        super().__init__()

        # Hyperparameters
        self.alpha = alpha
        self.UNK = unk_token

        # Term Probabilities
        self.community = community
        self.term_probability = self._read_term_probability_matrix(term_prob_csv)

        # Word Vectors
        self.wv = KeyedVectors.load(wv_save_path)
        self.vocab = list(self.wv.key_to_index.keys())
        self.num_vocab = len(self.vocab)
        self.embedding_dim = self.wv.vector_size
        

    def build(self, jsonl_path: str, content_field: str, id_field: str) -> None:
        sentences, ids =  super().build(jsonl_path, content_field, id_field)
        sentences = [re.findall('\w+', sentence) for sentence in sentences]
        self.ids = ids

        self.matrix = np.zeros((len(sentences) , self.embedding_dim))

        for i, sentence in enumerate(sentences) : 
            self.matrix[i , :] = self._get_sentence_embedding(sentence)

    def fit(self, texts: list) -> np.array:
        
        text_repr = []

        for text in texts : 
            text_repr.append(self._get_sentence_embedding(text))

        return np.array(text_repr)



    def _get_word_embedding(self, word) : 
        
        if word in self.vocab : 
            return self.wv[word]
            
        return self.wv[self.UNK]

    def _get_word_probability(self, word) : 

        if word in self.term_probability.keys() : 
            return self.term_probability[word]
        
        else : 
            return 0.0


    def _get_sentence_embedding(self, sentence) : 

        sentence_embedding = np.zeros((self.embedding_dim , ))
        num_words = 0

        for word in sentence : 

            word_is_valid , word = self._process_word(word)

            if word_is_valid : 

                vec = self._get_word_embedding(word)
                term_prob = self._get_word_probability(word)
                num_words += 1

                sentence_embedding += self.alpha * (1/(1-term_prob)) * vec

        if num_words == 0 : 
            return sentence_embedding

        sentence_embedding /= num_words

        return sentence_embedding 



    def _read_term_probability_matrix(self, term_probability_matrix_path) : 
        df = pd.read_csv(term_probability_matrix_path) 
        return  df[self.community].to_dict()

    def _process_word(self, word) : 
        
        processed_word = word.strip().lower()
        if len(processed_word) > 1 : 
            word_is_valid = True
        else : 
            word_is_valid = False

        return word_is_valid , processed_word



if __name__ == '__main__' : 

    from src.experiments import counts, embeddings

    wv_file = 'data/airbnb_hosts.wordvectors'
    term_prob_file = 'data/term_probability_matrix.csv'
    reddit_jsonl_file = 'data/airbnb_hosts.jsonl'


    domain_sim = DomainSim(wv_file, term_prob_file, 'airbnb')
    domain_sim.build(reddit_jsonl_file, 'body' , 'body')

    print(domain_sim.matrix)
    print(domain_sim.ids[:10])

    import json
    with open('data/labels.json') as f : 
        data = json.load(f)
        print('Finished loading data.')
        airbnb_data = data[1]['quotes']
        print('Airbnb Data : \n' , airbnb_data)
    

    airbnb_data_repr = domain_sim.fit(airbnb_data)
    print(airbnb_data_repr)
    print(domain_sim.similarity(airbnb_data_repr).shape)

    import pandas as pd 

    df = pd.DataFrame(domain_sim.similarity(airbnb_data_repr), index=domain_sim.ids, columns=airbnb_data)
    df.to_csv('domain_similarity.csv')



