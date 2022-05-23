from src.similarities.similarity import Sim
from sentence_transformers import SentenceTransformer
import numpy as np

class SBERTSim(Sim) : 

    def __init__(self, model_name:str='paraphrase-MiniLM-L3-v2') -> None:
        super().__init__()
        self.model = SentenceTransformer(model_name)

    def build(self, jsonl_path: str, content_field: str, id_field:str) -> None:
        content, ids = super().build(jsonl_path, content_field, id_field)
        print('len(content) : ' , len(content))
        self.matrix = self.model.encode(content)
        self.ids = ids

    def fit(self, texts: list) -> np.array:
        return self.model.encode(texts)

    def similarity(self, text_repr: np.array) -> float:
        return super().similarity(text_repr)
        

if __name__ == '__main__' : 

    sbert_sim = SBERTSim()
    sbert_sim.build('data/airbnb_hosts.jsonl', 'body', 'body')
    print(sbert_sim.matrix.shape)
    
    import json
    with open('data/labels.json') as f : 
        data = json.load(f)
        print('Finished loading data.')
        airbnb_data = data[1]['quotes']

    airbnb_data_repr = sbert_sim.fit(airbnb_data)

    print(sbert_sim.matrix.shape)
    print(airbnb_data_repr)

    print(sbert_sim.similarity(airbnb_data_repr).shape)


    import pandas as pd 

    df = pd.DataFrame(sbert_sim.similarity(airbnb_data_repr), index=sbert_sim.ids, columns=airbnb_data)
    df.to_csv('sbert_similarity.csv')