"""
Classification, calibration, and betting metrics.
"""

from typing import Dict, Optional
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.calibration import calibration_curve


class MetricsCalculator:
    def classification(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        return {
            "log_loss": log_loss(y_true, y_proba),
            "accuracy": accuracy_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def calibration(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> Dict:
        # brier score
        brier = np.mean((y_proba - y_true) ** 2)
        # expected calibration error (ECE)
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(y_proba, bins) - 1, 0, n_bins - 1)
        ece = 0.0
        for i in range(n_bins):
            m = idx == i
            if m.sum() > 0:
                ece += (m.sum() / len(y_true)) * abs(y_true[m].mean() - y_proba[m].mean())
        return {"brier_score": float(brier), "expected_calibration_error": float(ece)}

