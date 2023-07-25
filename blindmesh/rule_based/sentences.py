import random
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SentenceFilter(TransformerMixin, BaseEstimator):
    """Sklearn component that filters out certain sentences given some attributes
    of them. Texts that are passed to the component have to be sentencized and
    tokenized already.
    All parameters are optional, if nothing is passed into the constructor
    it will act as the identity.

    Parameters
    ----------
    length_range: tuple, default (0, None)
        A tuple of the lower and upper boundary on sentence length.
        If an integer it will be interpreted as absolute character count.
        If a float it will be interpreted as a quantile, and will be learned
        from data using reservoir sampling.
    max_memory: int, default 5000
        Maximum size of the reservoir sampled sentence length pool.
        This is what the lenght quantiles are calculated from.
        Lower number means worse estimate, but lower memory usage.
    """

    def __init__(
        self,
        length_range: tuple[Union[int, float], Union[int, float, None]] = (
            0,
            None,
        ),
        max_memory: int = 5000,
    ):
        self.length_range = length_range
        self.minl, self.maxl = length_range
        if isinstance(self.minl, int):
            self.min_length_quantile = False
            self.min_length = self.minl
        elif isinstance(self.minl, float):
            self.min_length_quantile = True
            self.min_length = 0
        else:
            raise ValueError("Minimum length either has to be float or int.")
        if isinstance(self.maxl, int):
            self.max_length_quantile = False
            self.max_length = self.maxl
        elif isinstance(self.maxl, float):
            self.max_length_quantile = True
            self.max_length = None
        elif self.maxl is None:
            self.max_length_quantile = False
            self.max_length = None
        else:
            raise ValueError("Minimum length either has to be float or int.")
        self.length_pool = []
        self.seen_lengths = 0
        self.max_memory = max_memory

    def append_reservoir(self, sentence: list[str]):
        """Adds token length to the pool with reservoir sampling."""
        if self.seen_lengths < self.max_memory:
            self.length_pool.append(len(sentence))
        else:
            j = random.randint(0, self.seen_lengths)
            if j < self.max_memory:
                self.length_pool[j] = len(sentence)
        self.seen_lengths += 1

    def fit(self, X: list[list[list[str]]], y=None):
        """Fits filter to the data.

        Parameters
        ----------
        X: list of list of list of str
            Documents sentencized and tokenized.

        Returns
        -------
        self
        """
        return self.partial_fit(X)

    def partial_fit(self, X: list[list[list[str]]], y=None):
        """Fits filter to the data.

        Parameters
        ----------
        X: list of list of list of str
            Documents sentencized and tokenized.

        Returns
        -------
        self
        """
        for doc in X:
            for sent in doc:
                self.append_reservoir(sent)
        if self.max_length_quantile:
            self.max_length = np.quantile(self.length_pool, self.maxl)  # type: ignore
        if self.min_length_quantile:
            self.min_length = np.quantile(self.length_pool, self.minl)  # type: ignore
        return self

    def passes(self, sentence: list[str]) -> bool:
        """Determines whether a sentence passes the filter or not."""
        sent_len = len(sentence)
        max_pass: bool = (self.max_length is None) or (
            sent_len <= self.max_length
        )
        min_pass: bool = sent_len >= self.min_length  # type: ignore
        return max_pass and min_pass

    def transform(self, X: list[list[list[str]]]) -> list[list[list[str]]]:
        """Filters out sentences based on the parameters of
        the component.

        Parameters
        ----------
        X: list of list of list of str
            Documents sentencized and tokenized.

        Returns
        -------
        list of list of list of str
            Documents with sentences that pass the filter.
        """
        res = []
        for doc in X:
            res_sents = [sent for sent in doc if self.passes(sent)]
            res.append(res_sents)
        return res
