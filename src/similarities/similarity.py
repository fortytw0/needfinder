import numpy as np
import json

class Sim(object) : 

    def __init__(self)->None: 
        self.matrix = None

    def build(self, jsonl_path:str, content_field:str) -> None : 
        content = []

        with open(jsonl_path) as f : 
            reddit_posts = f.readlines()
        
        for post in reddit_posts : 
            post = json.loads(post)
            if content_field in post : 
                content.append(post[content_field])

        return content

    def fit(self, texts:list) -> np.array : 
        pass

    def similarity(self, text_repr:np.array) -> float : 
        pass

        

