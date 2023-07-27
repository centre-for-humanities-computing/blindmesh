import warnings
from typing import Any, Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import maxabs_scale, minmax_scale


class DensityFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        density_estimator: BaseEstimator,
        positive_components: Iterable[Any] = (),
        negative_components: Iterable[Any] = (),
        positive_threshold: float = 0.5,
        negative_threshold: float = 0.5,
        freeze: bool = True,
    ):
        if not hasattr(density_estimator, "predict_proba"):
            raise TypeError(
                "Estimators passed to DensityFilter must have"
                " a predict_proba() method.\n"
                "Did you mean to use ComponentFilter?"
            )
        self.density_estimator = density_estimator
        self.positive_components = np.array(positive_components)
        self.negative_components = np.array(negative_components)
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.freeze = freeze
        self.negative_indices = None
        self.positive_indices = None

    def fit(self, X: list[str], y=None):
        if not self.freeze:
            self.density_estimator.fit(X, y)
        positive_indices = []
        negative_indices = []
        try:
            self.feature_names_in_ = (
                self.density_estimator.get_feature_names_out()
            )
            feature_to_index = {
                feature: index
                for index, feature in enumerate(self.feature_names_in_)
            }
            for component in self.positive_components:
                if isinstance(component, int) and (component >= 0):
                    positive_indices.append(component)
                else:
                    positive_indices.append(feature_to_index[component])
            for component in self.negative_components:
                if isinstance(component, int) and (component >= 0):
                    negative_indices.append(component)
                else:
                    negative_indices.append(feature_to_index[component])
        except AttributeError:
            positive_indices = self.positive_components
            negative_indices = self.negative_components
        self.positive_indices = np.array(positive_indices)
        self.negative_indices = np.array(negative_indices)
        return self

    def transform(self, X: list[str]) -> list[str]:
        probs = self.density_estimator.predict_proba(X)
        if len(probs.shape) == 1:
            passes = probs > self.positive_threshold
        elif len(probs.shape) == 2:
            if len(self.positive_components):
                positive_probs = probs[:, self.positive_indices].sum(axis=1)
            else:
                positive_probs = np.ones(probs.shape[0])
            if len(self.negative_components):
                negative_probs = probs[:, self.negative_indices].sum(axis=1)
            else:
                negative_probs = np.zeros(probs.shape[0])
            passes = (positive_probs >= self.positive_threshold) & (
                negative_probs <= self.negative_threshold
            )
        else:
            raise TypeError(
                "The probability estimator is expected to give"
                " probabilities for each feature or one probabilty"
                " per document. The encountered probabilities were"
                " a tensor of higher rank."
            )
        return [text for p, text in zip(passes, X) if p]


class ComponentFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator: TransformerMixin,
        positive_components: Iterable[Any] = (),
        negative_components: Iterable[Any] = (),
        positive_threshold: float = 0.5,
        negative_threshold: float = 0.5,
        as_probability: bool = True,
        freeze: bool = True,
    ):
        if not hasattr(estimator, "transform"):
            raise TypeError(
                "Estimators passed to ComponentFilter must have"
                " a transformed() method.\n"
                "Did you mean to use DensityFilter?"
            )
        self.estimator = estimator
        self.positive_components = np.array(positive_components)
        self.negative_components = np.array(negative_components)
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.as_probability = True
        self.freeze = freeze
        self.negative_indices = None
        self.positive_indices = None
        self.as_probability = as_probability

    def fit(self, X: list[str], y=None):
        if not self.freeze:
            self.estimator.fit(X, y)
        positive_indices = []
        negative_indices = []
        try:
            self.feature_names_in_ = self.estimator.get_feature_names_out()
            feature_to_index = {
                feature: index
                for index, feature in enumerate(self.feature_names_in_)
            }
            for component in self.positive_components:
                if isinstance(component, int) and (component >= 0):
                    positive_indices.append(component)
                else:
                    positive_indices.append(feature_to_index[component])
            for component in self.negative_components:
                if isinstance(component, int) and (component >= 0):
                    negative_indices.append(component)
                else:
                    negative_indices.append(feature_to_index[component])
        except AttributeError:
            positive_indices = self.positive_components
            negative_indices = self.negative_components
        self.positive_indices = np.array(positive_indices)
        self.negative_indices = np.array(negative_indices)
        return self

    def transform(self, X: list[str]) -> list[str]:
        transformed = self.estimator.transform(X)
        if len(transformed.shape) == 1:
            passes = transformed > self.positive_threshold
        elif len(transformed.shape) == 2:
            if self.as_probability:
                if np.any(transformed < 0):
                    transformed = minmax_scale(transformed, axis=1)
                else:
                    transformed = maxabs_scale(transformed, axis=1)
            if len(self.positive_components):
                positive = transformed[:, self.positive_indices].sum(axis=1)
            else:
                positive = np.full(
                    transformed.shape[0], self.positive_threshold
                )
            if len(self.negative_components):
                negative = transformed[:, self.negative_indices].sum(axis=1)
            else:
                negative = np.full(transformed.shape[0], self.negative_indices)
            passes = (positive >= self.positive_threshold) & (
                negative <= self.negative_threshold
            )
        else:
            raise TypeError(
                "The estimator is expected to give"
                " importances for each component or one score"
                " per document. The encountered scores were"
                " a tensor of higher rank."
            )
        return [text for p, text in zip(passes, X) if p]
