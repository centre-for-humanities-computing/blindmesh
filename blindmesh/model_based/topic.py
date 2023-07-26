from typing import Any, Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import maxabs_scale, minmax_scale


def infer_topic_names(pipeline: Pipeline, top_n: int = 4) -> list[str]:
    """Infers names of topics from a trained topic model's components.
    This method does not take empirical counts or relevance into account, therefore
    automatically assigned topic names can be of low quality.

    Parameters
    ----------
    pipeline: Pipeline
        Sklearn compatible topic pipeline.
    top_n: int, default 4
        Number of words used to name the topic.

    Returns
    -------
    list of str
        List of topic names.
    """
    _, vectorizer = pipeline.steps[0]
    _, topic_model = pipeline.steps[-1]
    components = topic_model.components_
    vocab = vectorizer.get_feature_names_out()
    highest = np.argpartition(-components, top_n)[:, :top_n]
    top_words = vocab[highest]
    topic_names = []
    for i_topic, words in enumerate(top_words):
        name = "_".join(words)
        topic_names.append(f"{i_topic}_{name}")
    return topic_names


class TopicClassifier(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, topic_pipeline: Pipeline, freeze: bool = False):
        self.freeze = freeze
        self.topic_pipeline = topic_pipeline

    def fit(self, X: list[str], y=None):
        if not self.freeze:
            self.topic_pipeline.fit(X, y)
        self.classes_ = np.array(
            infer_topic_names(self.topic_pipeline, top_n=4)
        )
        return self

    def transform(self, X: list[str]) -> ArrayLike:
        return self.topic_pipeline.transform(X)

    def predict_proba(self, X: list[str]) -> ArrayLike:
        transformed = self.topic_pipeline.transform(X)
        if np.all(transformed >= 0):
            probs = maxabs_scale(transformed, axis=1)
        else:
            probs = minmax_scale(transformed, axis=1)
        return probs

    def get_feature_names_out(self) -> np.ndarray:
        if self.classes_ is None:
            raise NotFittedError("Topic names have not been inferred yet.")
        return self.classes_

    def predict(self, X: list[str]) -> ArrayLike:
        transformed = self.topic_pipeline.transform(X)
        i_label = np.argmax(transformed, axis=1)
        label = self.classes_[i_label]
        return label
