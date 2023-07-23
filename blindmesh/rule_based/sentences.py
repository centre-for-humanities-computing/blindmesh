import random
from collections import Counter
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SentenceFilter(TransformerMixin, BaseEstimator):
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
            self.min_length_quantile = False
            self.min_length = 0
        else:
            raise ValueError("Minimum length either has to be float or int.")
        if isinstance(self.maxl, int):
            self.max_length_quantile = False
            self.max_length = self.minl
        elif isinstance(self.maxl, float):
            self.max_length_quantile = False
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
        if self.seen_lengths < self.max_memory:
            self.length_pool.append(len(sentence))
        else:
            j = random.randint(0, self.seen_lengths)
            if j < self.max_memory:
                self.length_pool[j] = len(sentence)
        self.seen_lengths += 1

    def partial_fit(self, X: list[list[list[str]]], y=None):
        for doc in X:
            for sent in doc:
                self.append_reservoir(sent)
        if self.max_length_quantile:
            self.max_length = np.quantile(self.length_pool, self.maxl)  # type: ignore
        if self.min_length_quantile:
            self.min_length = np.quantile(self.length_pool, self.minl)  # type: ignore

    def passes(self, sentence: list[str]) -> bool:
        sent_len = len(sentence)
        max_pass: bool = (self.max_length is None) or (
            sent_len <= self.max_length
        )
        min_pass: bool = sent_len >= self.min_length  # type: ignore
        return max_pass and min_pass

    def transform(self, X: list[list[list[str]]]) -> list[list[list[str]]]:
        res = []
        for doc in X:
            res_sents = []
            for sent in doc:
                res_tokens = [token for token in sent if self.passes(token)]
                res_sents.append(res_tokens)
            res.append(res_sents)
        return res
