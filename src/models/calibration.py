"""
Simple calibrators for probability outputs.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Platt scaling via logistic regression on model probabilities."""

    def __init__(self):
        self.lr = LogisticRegression(max_iter=1000)

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray):
        X = y_proba.reshape(-1, 1)
        self.lr.fit(X, y_true)
        return self

    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        X = y_proba.reshape(-1, 1)
        return self.lr.predict_proba(X)[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray):
        self.iso.fit(y_proba, y_true)
        return self

    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        return self.iso.transform(y_proba)

