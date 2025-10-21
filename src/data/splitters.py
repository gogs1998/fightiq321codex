"""
Temporal splitting and backtesting utilities with leakage prevention.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple

import pandas as pd
from loguru import logger


@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class TemporalSplitter:
    def __init__(self, val_start_date: str, test_start_date: str, date_column: str = "event_date"):
        self.val_start = pd.to_datetime(val_start_date)
        self.test_start = pd.to_datetime(test_start_date)
        self.date_column = date_column
        if self.val_start >= self.test_start:
            raise ValueError("val_start_date must be before test_start_date")

    def split(self, df: pd.DataFrame) -> DataSplit:
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df = df.sort_values(self.date_column).reset_index(drop=True)

        train_mask = df[self.date_column] < self.val_start
        val_mask = (df[self.date_column] >= self.val_start) & (df[self.date_column] < self.test_start)
        test_mask = df[self.date_column] >= self.test_start

        train = df[train_mask].reset_index(drop=True)
        val = df[val_mask].reset_index(drop=True)
        test = df[test_mask].reset_index(drop=True)

        self._validate_split(train, val, test)

        logger.info("Split Statistics:")
        logger.info(f"  Train: {len(train):,}")
        logger.info(f"  Val:   {len(val):,}")
        logger.info(f"  Test:  {len(test):,}")
        return DataSplit(train=train, val=val, test=test)

    def _validate_split(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        train_max = train[self.date_column].max()
        val_min = val[self.date_column].min()
        val_max = val[self.date_column].max()
        test_min = test[self.date_column].min()
        if pd.notna(train_max) and pd.notna(val_min) and train_max >= val_min:
            raise ValueError(f"Temporal leakage: train_max ({train_max}) >= val_min ({val_min})")
        if pd.notna(val_max) and pd.notna(test_min) and val_max >= test_min:
            raise ValueError(f"Temporal leakage: val_max ({val_max}) >= test_min ({test_min})")


class WalkForwardSplitter:
    def __init__(
        self,
        initial_train_end: str,
        final_test_end: str,
        test_window_months: int = 3,
        step_months: int = 1,
        min_train_size: int = 1000,
        date_column: str = "event_date",
    ):
        self.initial_train_end = pd.to_datetime(initial_train_end)
        self.final_test_end = pd.to_datetime(final_test_end)
        self.test_window = timedelta(days=30 * test_window_months)
        self.step = timedelta(days=30 * step_months)
        self.min_train_size = min_train_size
        self.date_column = date_column

    def create_folds(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df = df.sort_values(self.date_column).reset_index(drop=True)

        folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        current_train_end = self.initial_train_end

        while True:
            test_start = current_train_end
            test_end = test_start + self.test_window
            if test_start >= self.final_test_end:
                break

            train_mask = df[self.date_column] < test_start
            test_mask = (df[self.date_column] >= test_start) & (df[self.date_column] < test_end)

            train_df = df[train_mask].reset_index(drop=True)
            test_df = df[test_mask].reset_index(drop=True)

            if len(train_df) >= self.min_train_size and len(test_df) > 0:
                folds.append((train_df, test_df))

            current_train_end += self.step

        logger.info(f"Created {len(folds)} walk-forward folds")
        return folds

