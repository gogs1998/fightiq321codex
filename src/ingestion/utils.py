from __future__ import annotations

from typing import List
import pandas as pd


def dedupe_by_keys(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    if not key_cols:
        return df
    return df.drop_duplicates(subset=key_cols, keep="last")

