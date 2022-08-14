import os
from sentence_transformers import SentenceTransformer

from typing import List
import numpy as np

from src.experiment2.models.model import Model

class SBERT(Model) : 

    def __init__(self, 
                model_name:str='multi-qa-mpnet-base-dot-v1',  
                model_save_path:str='models',
                **kwargs) -> None:

        os.environ['SENTENCE_TRANSFORMERS_CACHE'] = model_save_path

        self.model = SentenceTransformer(model_name)
        
        
    def encode(self, sentences: List[str]) -> np.ndarray:
        return self.model.encode(sentences)