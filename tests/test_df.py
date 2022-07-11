import unittest
from src.corpus import Corpus


class TestStringMethods(unittest.TestCase):

    def test_isupper(self):
        corpus = Corpus(["tests/fixtures/twodocs.jsonl"])
        # corpus = ["hello world", "goodbye world"]
        for word in zip(corpus.vocab):
            if word == "hello":
                self.assertTrue(df, 1)
            if word == "goodbye":
                self.assertTrue(df, 1)
            if word == "world":
                self.assertTrue(df, 2)


if __name__ == '__main__':
    unittest.main()
