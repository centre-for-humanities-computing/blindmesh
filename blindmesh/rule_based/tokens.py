import random
from collections import Counter
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TokenFilter(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        negative: list[str],
        length_range: tuple[Union[int, float], Union[int, float, None]] = (
            0,
            None,
        ),
        frequency_range: tuple[float, float] = (0.0, 1.0),
        max_memory: int = 5000,
    ):
        self.negative = set(negative)
        self.length_range = length_range
        self.frequency_range = frequency_range
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
        self.frequencies = Counter()

    def append_reservoir(self, token: str):
        if self.seen_lengths < self.max_memory:
            self.length_pool.append(len(token))
        else:
            j = random.randint(0, self.seen_lengths)
            if j < self.max_memory:
                self.length_pool[j] = len(token)
        self.seen_lengths += 1

    def partial_fit(self, X: list[list[list[str]]], y=None):
        tokens = []
        for doc in X:
            for sent in doc:
                for token in sent:
                    tokens.append(token)
                    self.append_reservoir(token)
        if self.max_length_quantile:
            self.max_length = np.quantile(self.length_pool, self.maxl)  # type: ignore
        if self.min_length_quantile:
            self.min_length = np.quantile(self.length_pool, self.minl)  # type: ignore
        self.frequencies.update(tokens)

    def passes(self, token: str) -> bool:
        token_length = len(token)
        token_frequency = self.frequencies.get(token, 0)
        max_pass: bool = (self.max_length is None) or (
            token_length <= self.max_length
        )
        min_pass: bool = token_length >= self.min_length  # type: ignore
        freq_pass: bool = (token_frequency >= self.frequency_range[0]) and (
            token_frequency <= self.frequency_range[1]
        )
        return (
            (token not in self.negative)
            and max_pass
            and min_pass
            and freq_pass
        )

    def transform(self, X: list[list[list[str]]]) -> list[list[list[str]]]:
        res = []
        for doc in X:
            res_sents = []
            for sent in doc:
                res_tokens = [token for token in sent if self.passes(token)]
                res_sents.append(res_tokens)
            res.append(res_sents)
        return res
