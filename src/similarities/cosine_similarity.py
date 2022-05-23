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


    def build(self, jsonl_path: str, content_field: str, id_field:str) -> None:

        content, ids = super().build(jsonl_path, content_field, id_field)
        self.vectorizer = TfidfVectorizer()
        self.ids = ids
        tfidf = self.vectorizer.fit_transform(content)
        self.matrix = tfidf.toarray()


    def fit(self, texts: list) -> np.array:        
        return self.vectorizer.transform(texts)

    def similarity(self, text_repr: np.array) -> float:
        return super().similarity(text_repr)





if __name__ == '__main__' : 

    tfidf_sim = TFIDFSim()
    tfidf_sim.build('data/airbnb_hosts.jsonl', 'body', 'body')
    print(tfidf_sim.matrix.shape)

    import json
    with open('data/labels.json') as f : 
        data = json.load(f)
        print('Finished loading data.')
        airbnb_data = data[1]['quotes']
        print('Airbnb Data : \n' , airbnb_data)

    airbnb_data_repr = tfidf_sim.fit(airbnb_data)

    print(tfidf_sim.similarity(airbnb_data_repr))

    import pandas as pd 

    df = pd.DataFrame(tfidf_sim.similarity(airbnb_data_repr), index=tfidf_sim.ids, columns=airbnb_data)
    df.to_csv('cosine_similarity.csv')





