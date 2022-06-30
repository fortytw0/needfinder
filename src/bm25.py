from typing import List
from rank_bm25 import BM25Okapi
# https://pypi.org/project/rank-bm25/
from src.corpus import Corpus
import numpy as np

# python -m src.bm25

class WhitespaceTokenizer(object):
    def __init__(self):
        pass

    # assume python 3.9 AH https://stackoverflow.com/questions/52623204/how-to-specify-method-return-type-list-of-what-in-python
    def tokenize(self, document: str) -> List[str]: 
        return document.split(" ")

class BM25Prefetcher(object) : 

    def __init__(self, corpus, tokenizer = WhitespaceTokenizer()) -> None:
     
        self.corpus = corpus
        self.tokenizer = tokenizer
        tokenized_corpus = [self.tokenizer.tokenize(doc) for doc in corpus.data]
        index = BM25Okapi(tokenized_corpus)
        self.index = index

    def query_top_K(self, query: str, K=1000) -> list[str]:
        tokenized_query = self.tokenizer.tokenize(query)
        doc_scores = self.index.get_scores(tokenized_query)
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        # I find the syntax below confusing but seems to work in informal tests
        top_k_ix = np.argpartition(doc_scores, -K)[-K:] 
        top_docs = list(self.corpus.data[i] for i in top_k_ix)
        return top_docs

if __name__ == '__main__' : 

    corpus = Corpus(['tests/fixtures/airbnb1k.jsonl'])
    prefetcher = BM25Prefetcher(corpus)
    top_K = prefetcher.query_top_K("there are hidden security cameras gross", 5)
    print(top_K)


