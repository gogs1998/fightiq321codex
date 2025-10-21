from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import pandas as pd


class ParquetSink:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_dataframe(
        self,
        table: str,
        df: pd.DataFrame,
        key_cols: Optional[List[str]] = None,
        replace: bool = False,
    ) -> Path:
        path = self.base_dir / f"{table}.parquet"
        if replace or not path.exists():
            combined = df.copy()
        else:
            try:
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, df], axis=0, ignore_index=True)
            except Exception:
                combined = df.copy()

        if key_cols:
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
            combined = combined.dropna(subset=key_cols)
            for col in key_cols:
                if combined[col].dtype == object:
                    combined = combined[~combined[col].astype(str).str.lower().isin(["nan", "none", ""])]
        combined.to_parquet(path, index=False)
        return path


class BigQuerySink:
    def __init__(self, project_id: str, dataset: str):
        try:
            from google.cloud import bigquery  # type: ignore
        except Exception as e:
            raise RuntimeError("google-cloud-bigquery is required for BigQuerySink") from e
        self.project_id = project_id
        self.dataset = dataset
        self.client = bigquery.Client(project=project_id)

    def write_dataframe(self, table: str, df: pd.DataFrame, key_cols: Optional[List[str]] = None) -> None:
        from google.cloud import bigquery  # type: ignore
        table_ref = f"{self.project_id}.{self.dataset}.{table}"
        # naive upsert by key: load existing keys and filter
        if key_cols:
            cols = ", ".join(key_cols)
            query = f"SELECT {cols} FROM `{table_ref}`"
            try:
                existing = self.client.query(query).to_dataframe()
                if not existing.empty:
                    merge = df.merge(existing.drop_duplicates(), on=key_cols, how="left", indicator=True)
                    df = merge[merge["_merge"] == "left_only"][df.columns]
            except Exception:
                pass
        job = self.client.load_table_from_dataframe(df, table_ref)
        job.result()
        return None
