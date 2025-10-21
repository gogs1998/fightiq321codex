"""
Leak-safe UFC data loading and feature preparation.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import re
import numpy as np
import pandas as pd
from loguru import logger


class UFCDataLoader:
    """
    Load and prepare UFC data for modeling and inference.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

        self.target_cols = [
            "winner",
            "winner_encoded",
            "result",
            "result_details",
            "finish_round",
            "finish_time",
        ]

        self.metadata_cols = [
            "event_date",
            "fight_url",
            "event_name",
            "event_url",
            "referee",
            "event_city",
            "event_state",
            "event_country",
            "f_1_name",
            "f_2_name",
            "f_1_url",
            "f_2_url",
            "f_1_fighter_f_name",
            "f_1_fighter_l_name",
            "f_1_fighter_nickname",
            "f_2_fighter_f_name",
            "f_2_fighter_l_name",
            "f_2_fighter_nickname",
            "f_1_fighter_dob",
            "f_2_fighter_dob",
            "fighter_dob_f_1",
            "fighter_dob_f_2",
            "f_1_fighter_stance",
            "f_2_fighter_stance",
        ]

    def load_golden_dataset(self, golden_path: Optional[str] = None) -> pd.DataFrame:
        path = Path(golden_path) if golden_path else self.data_dir / "UFC_full_data_golden.csv"
        if not path.exists():
            raise FileNotFoundError(f"Golden dataset not found at {path}")
        logger.info(f"Loading golden dataset from {path}")
        df = pd.read_csv(path, parse_dates=["event_date"], low_memory=False)
        logger.info(f"Loaded {len(df):,} fights, {len(df.columns)} columns")
        return df

    def load_upcoming_fights(self, upcoming_path: str) -> pd.DataFrame:
        path = Path(upcoming_path)
        if not path.exists():
            raise FileNotFoundError(f"Upcoming fights file not found at {path}")
        logger.info(f"Loading upcoming fights from {path}")
        # event_date may be present or not; parse if present
        try:
            df = pd.read_csv(path, parse_dates=["event_date"], low_memory=False)
        except Exception:
            df = pd.read_csv(path, low_memory=False)
            if "event_date" in df.columns:
                df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        logger.info(f"Loaded {len(df):,} upcoming rows, {len(df.columns)} columns")
        return df

    @staticmethod
    def _is_current_fight_stat(column_name: str) -> bool:
        # Robust regex patterns to exclude current-fight stats
        leakage_patterns = [
            r"_r[1-5]_",  # round-by-round
            r"^r[1-5]_",
            r"r[1-5]_duration",
            r"f_[12]_total_strikes_(?:succ|att)$",
            r"f_[12]_sig_strikes_(?:succ|att)$",
            r"f_[12]_knockdowns$",
            r"f_[12]_submission_att$",
            r"f_[12]_ctrl_time_sec$",
            r"fight_duration",
            r"finish_round$",
            r"finish_time$",
            r"f_[12]_reversals$",
            r"f_[12]_td_[12]_(?:succ|att)$",
            r"f_[12]_takedown_(?:succ|att)$",
        ]
        for pattern in leakage_patterns:
            if re.search(pattern, column_name):
                return True
        return False

    def get_feature_columns(self, df: pd.DataFrame, exclude_odds: bool = False) -> List[str]:
        feature_cols = set(df.columns)
        feature_cols -= set(self.target_cols)
        feature_cols -= set(self.metadata_cols)

        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        feature_cols = feature_cols & numeric_cols

        # Remove current fight leakage
        feature_cols = {c for c in feature_cols if not self._is_current_fight_stat(c)}

        if exclude_odds:
            odds_kw = ["odds", "prob", "implied"]
            feature_cols = {c for c in feature_cols if not any(k in c.lower() for k in odds_kw)}

        feature_cols = sorted(list(feature_cols))
        logger.info(f"Using {len(feature_cols)} numeric features (leak-safe)")
        return feature_cols

    def prepare_features_target(
        self,
        df: pd.DataFrame,
        target: str = "winner_encoded",
        exclude_odds: bool = False,
        remove_draws: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        features = self.get_feature_columns(df, exclude_odds=exclude_odds)
        X = df[features].copy()
        y = df[target].copy() if target in df.columns else pd.Series([], dtype=float)

        if remove_draws and target == "winner_encoded" and target in df.columns:
            mask = y != -1
            X = X[mask]
            y = y[mask]

        # Do not impute here; defer to preprocessing strategy
        return X, y

    @staticmethod
    def align_to_training_features(X_new: pd.DataFrame, train_features: List[str]) -> pd.DataFrame:
        """Align columns and order to the training feature list; add missing columns as NaN."""
        return X_new.reindex(columns=train_features, fill_value=np.nan)
