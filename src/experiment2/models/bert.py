import os
from transformers import TFBertModel, BertTokenizer

from typing import List, Union
import numpy as np

from src.experiment2.models.model import Model

class BERT(Model) : 

    def __init__(self, 
                model_name:str='bert-base-uncased',  
                max_length:Union[None, int]=None,
                model_save_path:str='models',
                **kwargs) -> None:

        os.environ['TRANSFORMERS_CACHE'] = model_save_path

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertModel.from_pretrained(model_name)
        self.max_length = max_length
        

    def _tokenize(self, sentences:List[str]) : 

        if self.max_length is None : 
            return self.tokenizer(sentences,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='tf')

        else : 
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

    bert = BERT()
    
    outputs = bert.encode(sentences)
    print(outputs)
    print(outputs.shape)
        
