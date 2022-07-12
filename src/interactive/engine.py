import pandas as pd
import numpy as np
from rich.console import Console
from src.corpus import Corpus
from src.utils import WhitespaceTokenizer
from prompt_toolkit.completion import Completer, Completion, FuzzyCompleter
from prompt_toolkit.shortcuts import prompt


class BaseCompleter(Completer):

    def __init__(self, quote_path="data/interviewee_quotes.txt"):
        quotes = []
        with open(quote_path, "r") as inf:
            for i in inf:
                i = i.replace("\n", "")
                quotes.append(i)
        self.quotes = quotes

    def get_completions(self, document, complete_event):
        word = document.get_word_before_cursor()
        for quote in self.quotes:
            if quote.startswith(word):
                yield Completion(
                    quote,
                    start_position=-len(word),
                    style="fg:" + "green",
                    selected_style="fg:white bg:" + "green",
                )


class Renderer(object):
    def __init__(self, color="bold red"):
        self.color = color
        self.start = "[{}]".format(color)
        self.end = "[/{}]".format(color)

    def insert_markup_literal_match(self,
                                    input_string: str,
                                    bolded_words: list) -> str:
        '''
        This is a very simple function that will
        replace literal substrings, e.g "banana" in "bananas"
        '''
        for bolded_word in bolded_words:
            input_string = input_string.replace(
                bolded_word, self.start + bolded_word + self.end)
        return input_string

    def make_top_phrases_str(self, top_phrases):
        return ", ".join(top_phrases)

    def top_phrases_2_prompt(self, top_phrases: list[str]) -> str:

        top_phrases_str = self.make_top_phrases_str(top_phrases)
        out = self.insert_markup_literal_match(
             top_phrases_str,
             bolded_words=top_phrases)

        str_ = "\n" + "On reddit, they may discuss the following \
                       concepts related to this quote: " + out

        return str_

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
            output.append({"score": score,
                           "query_quote": query_quote,
                           "target_quote": target_quote})

        return output

    def get_top_K_with_substring_constraints(self,
                                             query_quote: str,
                                             constraints: list = [],
                                             K: int = 10) -> list[dict]:

        df = self.df
        for constraint in constraints:
            df = df[df[self.quote_column_name].apply(
                lambda x: constraint in x)]

        return self.get_top_K(query_quote, K, df)


def get_overlapping_words(query_quote: str,
                          target_quote: str,
                          min_length: int = 5,
                          tokenizer=WhitespaceTokenizer()) -> list[str]:
    query_quote_words = set(o for o in tokenizer.tokenize(query_quote))
    query_quote_words = set(
        o for o in query_quote_words if len(o) > min_length)
    target_quote_words = set(tokenizer.tokenize(target_quote))
    return query_quote_words & target_quote_words


class PhraseCountRanker(object):

    def __init__(self, corpus):

        self.corpus = corpus
        self.quote2ix = {quote: ix for ix, quote in enumerate(corpus.data)}

    def get_top_phrases_by_count(self,
                                 targets,
                                 top_k_threshold: int = 25,
                                 min_threshold: int = 1):
        '''
        return the top phrases in some posts.
        The posts are "targets" and are entered as a list
        '''

        ixs = [self.quote2ix[quote] for quote in targets]

        phrases_in_top = self.corpus.data_phrases_counts[ixs, :]
        phrase_counts = np.asarray(phrases_in_top.sum(axis=0))[0]

        top_phrases_ix = np.argpartition(
            phrase_counts, -top_k_threshold)[-top_k_threshold:]

        top_phrases = []
        for ix in top_phrases_ix:
            phrase = self.corpus.phrase_vocab[ix]
            if phrase_counts[ix] > min_threshold:
                top_phrases.append(phrase)
        return top_phrases


if __name__ == "__main__":

    engine = QueryEngine("data/results/arora_sim.csv")

    renderer = Renderer()

    console = Console(highlighter=None)

    corpus = Corpus(["data/airbnb_hosts.phrases.jsonl"], phrases=True)

    quote = prompt("Pick a quote: ", completer=FuzzyCompleter(BaseCompleter()))

    top_k = engine.get_top_K(quote)

    targets = [o["target_quote"] for o in top_k]

    phrase_count_ranker = PhraseCountRanker(corpus)

    ### Figure out what the top phrases are

    top_phrases = phrase_count_ranker.get_top_phrases_by_count(targets)

    console.print(renderer.top_phrases_2_prompt(top_phrases))

    with open("data/phrases.txt", "w") as of:
        of.write("\n".join(top_phrases + ["no"]))

    top_phrases_str = renderer.make_top_phrases_str(top_phrases)
    console.print(
        f"Do you want to investigate any of the following: {top_phrases_str}?")

    completer = FuzzyCompleter(BaseCompleter("data/phrases.txt"))

    ### Ask if they want to investigate a phrase

    phrase = prompt("\n Type the phrase you \
                    want to invesigate, or type no .. ",
                    completer=completer)

    lexical_requirements = [phrase]

    if phrase != "no":
        top_k = engine.get_top_K_with_substring_constraints(
            quote, lexical_requirements)

    for k in top_k:
        lexical_matches = get_overlapping_words(
            k["query_quote"], k["target_quote"])

        if phrase != "no":
            lexical_matches = lexical_requirements
        else:
            lexical_matches = lexical_matches

        out = renderer.insert_markup_literal_match(
            k['query_quote'], bolded_words=lexical_matches)
        console.print("\nChi: " + out.replace("\n", ""), style="white")
        out = renderer.insert_markup_literal_match(
            k['target_quote'], bolded_words=lexical_matches)
        console.print("Reddit: " + out.replace("\n", ""), style="white")
