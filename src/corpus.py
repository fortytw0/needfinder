from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm as tqdm
import numpy as np
import pandas as pd
from src.utils import read_jsonl

from typing import List


class Corpus(object):

    def __init__(self,
                 corpus_files: List[str],
                 content_field: str = 'body',
                 maxlines: float = 1e5,
                 phrases: bool = False,
                 ) -> None:

        self.corpus_files = corpus_files
        self.data = []

        for corpus_file in self.corpus_files:
            self.data.extend(read_jsonl(
                corpus_file, field=content_field, max_lines=maxlines))

        if phrases:
            import phrasemachine
            data_phrases = []
            for post in tqdm(self.data):
                post_phrases = phrasemachine.get_phrases(post)["counts"]
                post_phrases = list(post_phrases.keys())
                data_phrases.append({"phrases": post_phrases})
            self.data_phrases = pd.DataFrame(data_phrases, index=self.data)

        # a binary count of document frequencies
        # see tests/test_df and tests/fixtures/twodocs
        doc_frequency_vectorizer = CountVectorizer(binary=True)
        dfs = doc_frequency_vectorizer.fit_transform(self.data).sum(axis=0)
        self.dfs = np.asarray(dfs)[0]

        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(self.data).sum(axis=0).getA1()

        self.total_words = np.sum(X)
        self.counts = {word: count for word, count in zip(
            self.vectorizer.get_feature_names_out(), X)}
        self.vocab = self.vectorizer.get_feature_names_out()
        self.vocab_size = len(self.vocab)
        self.word_probs = {
            word: count/self.total_words for word, count in self.counts.items()}


if __name__ == '__main__':

    corpus = Corpus(['data/airbnb_hosts.jsonl'], phrases=True)
    singletons = []
    for w, wc in corpus.counts.items():

        print(w)
        print(wc)

        if wc <= 1:

            singletons.append(w)

    print('Signletons : ')
    for word in singletons:
        print('\t', word)

    print('---')
    print('Number of singletons : ', len(singletons))
