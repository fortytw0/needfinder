import unittest
from src.corpus import Corpus


class TestStringMethods(unittest.TestCase):

    def test_dfs(self):
        corpus = Corpus(["tests/fixtures/twodocs.jsonl"])
        dfs = corpus.dfs
        # corpus = ["hello world", "goodbye world"]
        for word in zip(corpus.vocab):
            ix = corpus.vectorizer.vocabulary_[word[0]]
            if word[0] == "hello":
                self.assertTrue(dfs[ix], 1)
            if word[0] == "goodbye":
                self.assertTrue(dfs[ix], 1)
            if word[0] == "world":
                self.assertTrue(dfs[ix], 2)

    def test_phrases_dfs(self):
        corpus = Corpus(["tests/fixtures/demo.phrases.jsonl"], phrases=True)
        dfs = corpus.phrase_dfs
        for word in zip(corpus.phrase_vocab):
            ix = corpus.phrase_vectorizer.vocabulary_[word[0]]
            if word[0] == "red car":
                self.assertTrue(dfs[ix], 2)
            if word[0] == "blue car":
                self.assertTrue(dfs[ix], 1)


if __name__ == '__main__':
    unittest.main()
