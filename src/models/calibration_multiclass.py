from __future__ import annotations

"""
Simple calibration helpers:
- WinnerCalibrator: Platt scaling (logistic regression on proba) per segment.
- TemperatureScaling: global temperature applied to logits for multiclass.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class WinnerPlatt:
    def __init__(self):
        self.lr = LogisticRegression(max_iter=1000)

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray):
        X = y_proba.reshape(-1, 1)
        self.lr.fit(X, y_true)
        return self

    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        X = y_proba.reshape(-1, 1)
        return self.lr.predict_proba(X)[:, 1]


class TemperatureScaling:
    def __init__(self, T: float | None = None):
        self.T = T if T is not None else 1.0

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, logits: np.ndarray, y_true_codes: np.ndarray, grid: list[float] | None = None):
        # Grid search temperature to minimize NLL
        if grid is None:
            grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
        best_T, best_nll = 1.0, 1e9
        for T in grid:
            p = self._softmax(logits / T)
            # negative log likelihood
            idx = (np.arange(len(p)), y_true_codes)
            eps = 1e-12
            nll = -np.mean(np.log(p[idx] + eps))
            if nll < best_nll:
                best_nll, best_T = nll, T
        self.T = best_T
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return self._softmax(logits / self.T)

