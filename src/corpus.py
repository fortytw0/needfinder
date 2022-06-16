from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils import read_jsonl

from typing import List


class Corpus(object) : 

    def __init__(self, 
                corpus_files:List[str], 
                content_field:str='body',
                maxlines:float=1e5,
                ) -> None:


        
        self.corpus_files = corpus_files
        self.data = []

        for corpus_file in self.corpus_files : 
            self.data.extend(read_jsonl(corpus_file, field=content_field, max_lines=maxlines))


        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(self.data).sum(axis=0).getA1()
        
        print(X)
        print(type(X))
        print(X.shape)

        self.counts = {word:count for word,count in zip(self.vectorizer.get_feature_names_out(), X)}
        self.vocabs = self.vectorizer.get_feature_names_out()


if __name__ == '__main__' : 


    corpus = Corpus(['data/airbnb_hosts.jsonl'])
    singletons = []
    for w, wc in corpus.counts.items() : 
        
        print(w)
        print(wc)
        
        if wc <= 1 : 
            
            singletons.append(w)

    print('Signletons : ')
    for word in singletons : 
        print('\t' , word) 

    print('---')
    print('Number of singletons : ' , len(singletons))

    












        