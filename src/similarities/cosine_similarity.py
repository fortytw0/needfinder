from src.similarities.similarity import Sim

import pandas as pd
import numpy as np
import json


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Setup Logging 
import logging
logging.basicConfig(filename='logs/main.log',
                    level=logging.INFO, 
                    format='%(asctime)s :: %(levelname)s :: %(module)s :: %(message)s')
logging.info('Logging has been setup.')


class TFIDFSim(Sim) : 

    def __init__(self) -> None:
        super().__init__()
        self.vectorizer = None


    def build(self, jsonl_path: str, content_field: str) -> None:

        content = super().build(jsonl_path, content_field)
        self.vectorizer = TfidfVectorizer()
        tfidf = self.vectorizer.fit_transform(content)
        self.matrix = tfidf.toarray()


    def fit(self, jsonl_path:str, content_field:str='content') -> None:
        pass





if __name__ == '__main__' : 

    tfidf_sim = TFIDFSim()
    tfidf_sim.build('data/airbnb_hosts.jsonl', 'body')
    print(tfidf_sim.matrix.shape)





