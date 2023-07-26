import warnings
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ProbabilityFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        probability_estimator: BaseEstimator,
        positive_features: Iterable[str] = (),
        positive_feature_indices: Iterable[int] = (),
        threshold: float = 0.5,
        freeze: bool = True,
    ):
        self.probability_estimator = probability_estimator
        self.threshold = threshold
        self.positive_features = np.array(list(positive_features))
        self.positive_feature_indices = np.array(
            list(positive_feature_indices)
        )
        self.has_predict_proba = hasattr(
            probability_estimator, "predict_proba"
        )
        if not self.has_predict_proba:
            warnings.warn(
                "Probability estimator does not have"
                " predict_proba() method, transform() will"
                " be used to obtain unnormalized probabilities."
            )
        self.freeze = freeze

    def fit(self, X: list[str], y=None):
        if not self.freeze:
            self.probability_estimator.fit(X, y)
        if not len(self.positive_feature_indices) and len(
            self.positive_features
        ):
            try:
                self.classes_ = (
                    self.probability_estimator.get_feature_names_out()
                )
                self.n_features_in_ = len(self.classes_)
                class_to_index = dict(
                    zip(self.classes_, range(self.n_features_in_))
                )
                self.positive_feature_indices = np.array(
                    [class_to_index[c] for c in self.positive_features]
                )
            except AttributeError as e:
                raise ValueError(
                    "Could not obtain class names from prob estimator."
                ) from e
        if not len(self.positive_feature_indices):
            self.positive_feature_indices = np.array([0])
            warnings.warn(
                "No feature indices or names were specified."
                " We assume that either only one feature exists or"
                " that the zeroth feature is the positive one."
            )
        return self

    def transform(self, X: list[str]) -> list[str]:
        if self.has_predict_proba:
            probs = self.probability_estimator.predict_proba(X)
        else:
            probs = self.probability_estimator.transform(X)
        if len(probs.shape) == 1:
            self.n_features_in_ = 1
        elif len(probs.shape) == 2:
            self.n_features_in_ = probs.shape[1]
            positive_prob_sum = probs[:, self.positive_feature_indices].sum(
                axis=1
            )
            probs = positive_prob_sum / probs.sum(axis=1)
        else:
            raise TypeError(
                "The probability estimator is expected to give"
                " probabilities for each feature or one probabilty"
                " per document. The encountered probabilities were"
                " a tensor of higher rank."
            )
        passes = probs > self.threshold
        return [text for p, text in zip(passes, X) if p]
