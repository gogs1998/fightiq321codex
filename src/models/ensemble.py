"""
Stacking ensemble with out-of-fold predictions for time-series.
"""

from typing import Callable, Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit


class StackingEnsemble:
    def __init__(self, base_models: List[Dict], meta_model=None, n_splits: int = 5, random_state: int = 42):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(max_iter=1000, random_state=random_state)
        self.n_splits = n_splits
        self.base_models_final: Dict[str, any] = {}
        self.meta_model_fitted = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackingEnsemble":
        n = len(X)
        m = len(self.base_models)
        oof = np.zeros((n, m))
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for fold, (tr, va) in enumerate(tscv.split(X)):
            Xtr, ytr = X.iloc[tr], y.iloc[tr]
            Xva = X.iloc[va]
            for j, cfg in enumerate(self.base_models):
                model = cfg["trainer"](Xtr, ytr)
                oof[va, j] = _predict_proba(model, cfg["name"], Xva)

        self.meta_model_fitted = self.meta_model.fit(oof, y)

        # retrain on full data
        for cfg in self.base_models:
            self.base_models_final[cfg["name"]] = cfg["trainer"](X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base_preds = []
        for cfg in self.base_models:
            mdl = self.base_models_final[cfg["name"]]
            base_preds.append(_predict_proba(mdl, cfg["name"], X))
        Xmeta = np.column_stack(base_preds)
        return self.meta_model_fitted.predict_proba(Xmeta)[:, 1]


def _predict_proba(model, name: str, X: pd.DataFrame) -> np.ndarray:
    lname = name.lower()
    if lname.startswith("xgb"):
        import xgboost as xgb

        return model.predict(xgb.DMatrix(X))
    if lname.startswith("lgb") or lname.startswith("lightgbm"):
        return model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    return model.predict_proba(X)[:, 1]

