#!/usr/bin/env python
"""
Demonstration of a custom completer wrapped in a `FuzzyCompleter` for fuzzy
matching.
"""
from prompt_toolkit.completion import Completer, Completion, FuzzyCompleter
from prompt_toolkit.shortcuts import CompleteStyle, prompt

class ColorCompleter(Completer):

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


def main():
    # Simple completion menu.

    prompt("Pick a quote: ", completer=FuzzyCompleter(ColorCompleter()))

    '''
    # Multi-column menu.
    prompt(
        "Type a color: ",
        completer=FuzzyCompleter(ColorCompleter()),
        complete_style=CompleteStyle.MULTI_COLUMN,
    )

    # Readline-like
    prompt(
        "Type a color: ",
        completer=FuzzyCompleter(ColorCompleter()),
        complete_style=CompleteStyle.READLINE_LIKE,
    )
    '''

if __name__ == "__main__":
    main()
