from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from transformers import pipeline


class ZeroShotClassifier(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self,
        classes: Iterable[str],
        model_name: str = "facebook/bart-large-mnli",
    ):
        self.classes_ = np.array(list(classes))
        self.n_classes = len(self.classes_)
        self.model_name = model_name
        self.pipe = pipeline(model=model_name)

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X: list[str]) -> np.ndarray:
        n_texts = len(X)
        res = np.empty((n_texts, self.n_classes))
        for i_doc, text in enumerate(X):
            out = self.pipe(text, candidate_labels=self.classes_)
            label_to_score = dict(zip(out["labels"], out["scores"]))
            for i_class, label in enumerate(self.classes_):
                res[i_doc, i_class] = label_to_score[label]
        return res

    def predict_proba(self, X: list[str]) -> np.ndarray:
        return self.transform(X)

    def predict(self, X: list[str]) -> np.ndarray:
        probs = self.transform(X)
        label_indices = np.argmax(probs, axis=1)
        return self.classes_[label_indices]

    def get_feature_names_out(self) -> np.ndarray:
        return self.classes_

    @property
    def class_to_index(self) -> dict[str, int]:
        return dict(zip(self.classes_, range(self.n_classes)))
