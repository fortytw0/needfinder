import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

class Sim(object) : 

    def __init__(self)->None: 
        self.matrix = None
        self.ids = None

    def build(self, jsonl_path:str, content_field:str, id_field:str) -> None : 
        content = []
        ids = []

        with open(jsonl_path) as f : 
            reddit_posts = f.readlines()
        
        for post in reddit_posts : 
            post = json.loads(post)
            if content_field in post : 
                content.append(post[content_field])
                ids.append(post[id_field])

        return content , ids

    def fit(self, texts:list) -> np.array : 
        pass

    def similarity(self, text_repr:np.array) -> float : 
        return cosine_similarity(self.matrix, text_repr)

        

