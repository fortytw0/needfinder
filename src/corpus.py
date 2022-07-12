from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm as tqdm
import numpy as np
import pandas as pd
from src.utils import read_jsonl
from src.utils import get_count_vectorizer_for_tokenized_lists

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
            self.data_phrases = []
            assert ".phrases" in corpus_file, f"you need to run $py scripts/add_phrases.py {corpus_file} first"
            for corpus_file in self.corpus_files:
                self.data_phrases.extend(read_jsonl(
                    corpus_file, field="phrases", max_lines=maxlines))

            self.phrase_vectorizer = get_count_vectorizer_for_tokenized_lists()

            assert len(self.data_phrases) > 0
            self.data_phrases_counts = self.phrase_vectorizer.fit_transform(self.data_phrases)

            # document frequencies for phrases
            self.phrase_dfs = np.asarray(self.data_phrases_counts.sum(axis=0))[0]

            self.phrase_vocab = self.phrase_vectorizer.get_feature_names_out()

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
            word: count/self.total_words
            for word, count in self.counts.items()}


if __name__ == '__main__':

    corpus = Corpus(['tests/fixtures/demo.phrases.jsonl'], phrases=True)
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
