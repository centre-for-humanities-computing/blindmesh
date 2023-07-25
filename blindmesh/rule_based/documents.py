import importlib
from typing import Iterable, Optional, Union

from sklearn.base import BaseEstimator, TransformerMixin

Num = Union[int, float]


def is_in_range(x: Num, r: tuple[Num, Optional[Num]]) -> bool:
    lower, upper = r
    lower_pass = x >= lower
    upper_pass = (upper is None) or (x <= upper)
    return upper_pass and lower_pass


class DocumentFilter(TransformerMixin, BaseEstimator):
    """Sklearn compatible component for quality filtering documents
    in a pipeline.

    Parameters
    ----------
    function_words: iterable of str or str
        Function words to count in documents.
        If a single string, it is interpreted as a language code,
        and function words are imported from spaCy.
        If its an iterable of strings, every token will be counted that's
        in the list.
    function_words_freq_range: tuple of (float, float), default (0.0, 1.0)
        Range of frequency of number of function words to accept.
        For example if you want minimally 10% of tokens to be function words,
        you should pass something like (0.1, 1.0)
    mean_token_length_range: tuple of (int, int or None), default (0, None)
        Range of mean token length to accept.
    mean_sentence_length_range: tuple of (int, int or None), default (0, None)
        Range of mean sentence length to accept.
    n_tokens_range: tuple of (int, int or None), default (0, None)
        Range of number of tokens to accept.
    n_sentences_range: tuple of (int, int or None), default (0, None)
        Range of number of sentences to accept.
    n_characters_range: tuple of (int, int or None), default (0, None)
        Range of number of characters to accept.
    """

    def __init__(
        self,
        function_words: Union[Iterable[str], str] = "en",
        function_words_freq_range: tuple[float, float] = (0.0, 1.0),
        mean_token_length_range: tuple[
            int,
            Optional[int],
        ] = (
            0,
            None,
        ),
        mean_sentence_length_range: tuple[int, Optional[int]] = (
            0,
            None,
        ),
        n_tokens_range: tuple[int, Optional[int]] = (
            0,
            None,
        ),
        n_sentences_range: tuple[int, Optional[int]] = (
            0,
            None,
        ),
        n_characters_range: tuple[int, Optional[int]] = (
            0,
            None,
        ),
    ):
        self.mean_token_length_range = mean_token_length_range
        self.mean_sentence_length_range = mean_sentence_length_range
        self.n_tokens_range = n_tokens_range
        self.n_sentences_range = n_sentences_range
        self.n_characters_range = n_characters_range
        self.function_words_freq_range = function_words_freq_range
        if isinstance(function_words, str):
            lang = function_words
            self.function_words = importlib.import_module(
                f"spacy.lang.{lang}.stop_words"
            ).STOP_WORDS
        elif function_words is None:
            self.function_words = set()
        else:
            self.function_words = set(function_words)

    def fit(self, X: list[list[list[str]]], y=None):
        """Does nothing exists for compatiblity."""
        return self

    def partial_fit(self, X: list[list[list[str]]], y=None):
        """Does nothing exists for compatiblity."""
        return self

    def passes(self, document: list[list[str]]) -> bool:
        """Determines whether a document should pass the filter or not."""
        n_tokens = 0
        n_function_words = 0
        n_sentences = len(document)
        n_characters = 0
        for sent in document:
            n_tokens += len(sent)
            for token in sent:
                n_characters += len(token)
                if token in self.function_words:
                    n_function_words += 1
        mean_token_length = n_characters / n_tokens if n_tokens else 0
        mean_sentence_length = n_tokens / n_sentences if n_sentences else 0
        function_word_freq = n_function_words / n_tokens if n_tokens else 0
        return (
            is_in_range(mean_token_length, self.mean_token_length_range)
            & is_in_range(
                mean_sentence_length, self.mean_sentence_length_range
            )
            & is_in_range(n_tokens, self.n_tokens_range)
            & is_in_range(n_sentences, self.n_sentences_range)
            & is_in_range(n_characters, self.n_characters_range)
            & is_in_range(function_word_freq, self.function_words_freq_range)
        )

    def transform(self, X: list[list[list[str]]]) -> list[list[list[str]]]:
        """Filters out documents based on the parameters of
        the component.

        Parameters
        ----------
        X: list of list of list of str
            Documents sentencized and tokenized.

        Returns
        -------
        list of list of list of str
            Documents that pass the filter.
        """
        return [doc for doc in X if self.passes(doc)]
