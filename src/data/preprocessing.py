"""
Feature-type-aware imputation with optional missingness indicators.
"""

from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from loguru import logger


class FeatureTypeImputationStrategy:
    def __init__(self, create_indicators: bool = True):
        self.create_indicators = create_indicators
        self.physical_imputer = SimpleImputer(strategy="median")
        self.career_imputer = SimpleImputer(strategy="median")
        self.odds_imputer = SimpleImputer(strategy="median")
        self.rolling_imputer = SimpleImputer(strategy="constant", fill_value=0)
        self.feature_groups: Dict[str, List[str]] = {
            "physical": [],
            "career": [],
            "rolling": [],
            "odds": [],
            "other": [],
        }
        self.indicator_cols: List[str] = []
        self.fitted = False

    def _categorize_features(self, columns: List[str]) -> Dict[str, List[str]]:
        groups = {"physical": [], "career": [], "rolling": [], "odds": [], "other": []}
        for col in columns:
            low = col.lower()
            if any(kw in low for kw in ["height", "reach", "weight", "age"]):
                groups["physical"].append(col)
            elif any(kw in low for kw in ["odds", "prob", "implied"]):
                groups["odds"].append(col)
            elif any(f"_{i}_" in col for i in range(3, 16)):
                groups["rolling"].append(col)
            elif any(
                kw in low
                for kw in [
                    "slpm",
                    "str_acc",
                    "sapm",
                    "str_def",
                    "td_avg",
                    "td_acc",
                    "td_def",
                    "sub_avg",
                    "fighter_w",
                    "fighter_l",
                    "fighter_d",
                ]
            ):
                groups["career"].append(col)
            else:
                groups["other"].append(col)
        return groups

    def fit(self, X: pd.DataFrame) -> "FeatureTypeImputationStrategy":
        self.feature_groups = self._categorize_features(X.columns.tolist())
        if len(self.feature_groups["physical"]) > 0:
            self.physical_imputer.fit(X[self.feature_groups["physical"]])
        if len(self.feature_groups["career"]) > 0:
            self.career_imputer.fit(X[self.feature_groups["career"]])
        if len(self.feature_groups["odds"]) > 0:
            self.odds_imputer.fit(X[self.feature_groups["odds"]])
        if len(self.feature_groups["rolling"]) > 0:
            self.rolling_imputer.fit(X[self.feature_groups["rolling"]])
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Imputer not fitted. Call fit() first.")
        X_imp = X.copy()
        missing_before = int(X_imp.isna().sum().sum())
        if len(self.feature_groups["physical"]) > 0:
            X_imp[self.feature_groups["physical"]] = self.physical_imputer.transform(
                X_imp[self.feature_groups["physical"]]
            )
        if len(self.feature_groups["career"]) > 0:
            X_imp[self.feature_groups["career"]] = self.career_imputer.transform(
                X_imp[self.feature_groups["career"]]
            )
        if len(self.feature_groups["odds"]) > 0:
            X_imp[self.feature_groups["odds"]] = self.odds_imputer.transform(
                X_imp[self.feature_groups["odds"]]
            )
        if len(self.feature_groups["rolling"]) > 0:
            X_imp[self.feature_groups["rolling"]] = self.rolling_imputer.transform(
                X_imp[self.feature_groups["rolling"]]
            )

        if self.create_indicators:
            indicator_cols = [col for col in X.columns if not col.endswith("_missing")]
            ind_df = pd.DataFrame(
                {f"{col}_missing": X[col].isna().astype(int) for col in indicator_cols},
                index=X.index,
            )
            existing_indicator_cols = [col for col in X_imp.columns if col.endswith("_missing")]
            if existing_indicator_cols:
                for col, series in ind_df.items():
                    X_imp[col] = series
                self.indicator_cols = list(ind_df.columns)
            else:
                self.indicator_cols = list(ind_df.columns)
                X_imp = pd.concat([X_imp, ind_df], axis=1)

        # Final safety: fill any remaining missing values
        missing_after = int(X_imp.isna().sum().sum())
        if missing_after > 0:
            X_imp = X_imp.fillna(0)
            missing_final = int(X_imp.isna().sum().sum())
            if missing_final > 0:
                logger.warning(f"{missing_final} missing values remain after final fillna(0)")
        else:
            logger.info(f"Imputed {missing_before} missing values")
        return X_imp

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)
