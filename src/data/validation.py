"""
Data integrity and sanity checks to guard against leakage and data errors.
"""

from typing import Dict, List
import pandas as pd
from loguru import logger


class DataValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> Dict:
        self._check_future_dates()
        self._check_missing_targets()
        self._check_impossible_values()
        self._check_odds_sanity()
        self._check_duplicates()
        return {
            "passed": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def _check_future_dates(self):
        if "event_date" not in self.df.columns:
            return
        today = pd.Timestamp.now()
        future = self.df[self.df["event_date"] > today]
        if len(future) > 0:
            self.warnings.append(
                f"Found {len(future)} future-dated rows; acceptable for upcoming predictions but review."
            )

    def _check_missing_targets(self):
        if "winner" in self.df.columns:
            missing = int(self.df["winner"].isna().sum())
            if missing > 0:
                self.errors.append(f"{missing} rows missing 'winner'.")
        if "winner_encoded" in self.df.columns:
            missing_e = int(self.df["winner_encoded"].isna().sum())
            if missing_e > 0:
                self.errors.append(f"{missing_e} rows missing 'winner_encoded'.")

    def _check_impossible_values(self):
        if "f_1_fighter_age" in self.df.columns:
            under = int((self.df["f_1_fighter_age"] < 18).sum())
            if under > 0:
                self.errors.append(f"{under} fighters under 18 (invalid for UFC).")
        if "f_1_fighter_height_cm" in self.df.columns:
            low = int((self.df["f_1_fighter_height_cm"] < 140).sum())
            high = int((self.df["f_1_fighter_height_cm"] > 230).sum())
            if low > 0:
                self.warnings.append(f"{low} heights < 140cm; review.")
            if high > 0:
                self.warnings.append(f"{high} heights > 230cm; review.")

    def _check_odds_sanity(self):
        if {"f_1_odds", "f_2_odds", "winner_encoded"}.issubset(self.df.columns):
            dfc = self.df.dropna(subset=["f_1_odds", "f_2_odds", "winner_encoded"])
            if len(dfc) == 0:
                return
            f1p = 1.0 / dfc["f_1_odds"].astype(float)
            f2p = 1.0 / dfc["f_2_odds"].astype(float)
            fav_is_f1 = f1p > f2p
            fav_won = (fav_is_f1 & (dfc["winner_encoded"] == 1)) | (~fav_is_f1 & (dfc["winner_encoded"] == 0))
            acc = float(fav_won.mean()) if len(fav_won) > 0 else 0.0
            if acc > 0.95:
                self.errors.append(
                    f"Favorite win rate {acc:.1%} too high; odds may be post-fight."
                )
            elif acc < 0.50:
                self.warnings.append(
                    f"Favorite win rate {acc:.1%} unusually low; verify odds integrity."
                )

    def _check_duplicates(self):
        if "fight_url" in self.df.columns:
            dups = int(self.df["fight_url"].duplicated().sum())
            if dups > 0:
                self.warnings.append(
                    f"Found {dups} duplicate fight_url entries; may cause leakage across splits."
                )

