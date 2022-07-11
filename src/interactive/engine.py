import pandas as pd
from rich import print
from rich.console import Console
from src.corpus import Corpus


class Renderer(object):
    def __init__(self, color="bold red"):
        self.color = color
        self.start = "[{}]".format(color)
        self.end = "[/{}]".format(color)

    def insert_markup_literal_match(self, input_string: str, bolded_words: list) -> str:
        '''
        This is a very simple function that will replace literal strings, e.g "banana" in "bananas"
        '''
        for bolded_word in bolded_words:
            input_string = input_string.replace(
                bolded_word, self.start + bolded_word + self.end)
        return input_string


class QueryEngine(object):

    def __init__(self, similarity_file, quote_column_name='Unnamed: 0'):
        self.similarity_file = similarity_file
        self.df = pd.read_csv(similarity_file)
        self.quote_column_name = quote_column_name

    def get_top_K(self, query_quote: str, K: int = 10, df=None) -> list[dict]:

        if df is None:
            df = self.df

        target_quotes = df.sort_values(by=[query_quote], ascending=False)
        target_quotes = target_quotes[0:K][self.quote_column_name].to_list()

        scores = df.sort_values(by=[query_quote], ascending=False)
        scores = scores[0:K][query_quote].to_list()

        output = []

        for score, target_quote in zip(scores, target_quotes):
            score = "{:.2%}".format(score)
            target_quote = target_quote.replace("\n", "")
            output.append({"score": score,
                           "query_quote": query_quote,
                           "target_quote": target_quote})

        return output

    def get_top_K_with_constraints(self, query_quote: str, constraints: list = [], K: int = 10) -> list[dict]:

        df = self.df
        for constraint in constraints:
            df = df[df[self.quote_column_name].apply(
                lambda x: constraint in x)]

        return self.get_top_K(query_quote, K, df)


def get_overlapping_words_simple(query_quote, target_quote):
    query_quote_words = [o for o in query_quote.split(" ") if len(o) > 5]
    target_quote_words = target_quote.split(" ")
    lexical_matches = set(query_quote_words) & set(target_quote_words)
    return list(lexical_matches)


if __name__ == "__main__":
    corpus = Corpus(["data/airbnb_hosts.jsonl"])
    engine = QueryEngine("data/results/arora_sim.csv")
    renderer = Renderer()
    console = Console(highlighter=None)

    lexical_requirements = ["clean"]

    for col in engine.df.columns[1:]:
        console.print("\n" + col, style="bold red")
        top_k = engine.get_top_K_with_constraints(col, lexical_requirements)
        for k in top_k:
            lexical_matches = get_overlapping_words_simple(
                k["query_quote"], k["target_quote"])
            lexical_matches = lexical_matches + lexical_requirements
            out = renderer.insert_markup_literal_match(
                k['query_quote'], bolded_words=lexical_matches)
            console.print("\nChi: " + out, style="white")
            out = renderer.insert_markup_literal_match(
                k['target_quote'], bolded_words=lexical_matches)
            console.print("Reddit: " + out, style="white")
