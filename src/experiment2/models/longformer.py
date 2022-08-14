import os
import numpy as np

from transformers import LongformerTokenizer, TFLongformerModel

from src.experiment2.models.model import Model
from typing import List, Union

class Longformer(Model) : 

    def __init__(self, 
                model_name:str='allenai/longformer-base-4096',  
                tokenizer_name:str='roberta-base',
                max_length:int=2048,
                model_save_path:str='models',
                **kwargs) -> None:

        os.environ['TRANSFORMERS_CACHE'] = model_save_path

        self.tokenizer = LongformerTokenizer.from_pretrained(tokenizer_name)
        self.model = TFLongformerModel.from_pretrained(model_name)
        self.max_length = max_length

    def _tokenize(self, sentences:List[str]) : 

        return self.tokenizer(sentences,
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors='tf')

    def encode(self, sentences: List[str]) -> np.ndarray:
        tokens = self._tokenize(sentences)
        encoded = self.model(tokens).last_hidden_state.numpy()
        mean_pool = np.average(encoded, axis=1)
        return mean_pool

if __name__ == '__main__' : 

    sentences = ['Hello World!' , 
                'Life the universe and Everything!' ,
                'Hey, whats going on?'
                ]

    longformer = Longformer()
    
    outputs = longformer.encode(sentences)
    print(outputs)
    print(outputs.shape)