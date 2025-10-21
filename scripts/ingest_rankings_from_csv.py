"""
Ingest historical rankings from CSV into raw rankings (parquet).
Source: FightIQ/data/UFC_rankings_history.csv
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.ingestion.sinks import ParquetSink


def main():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    csv_path = ROOT.parents[0] / "FightIQ" / "data" / "UFC_rankings_history.csv"
    if not csv_path.exists():
        logger.error(f"Rankings CSV not found: {csv_path}")
        sys.exit(2)
    df = pd.read_csv(csv_path)
    # Standardize
    df = df.rename(columns={'date':'rank_date','weightclass':'weight_class'})
    df['rank_date'] = pd.to_datetime(df['rank_date'], errors='coerce')
    df['fighter'] = df['fighter'].astype(str).str.strip()
    df['weight_class'] = df['weight_class'].astype(str).str.strip()
    if 'rank' in df.columns:
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
        zero_or_neg = (df['rank'] <= 0) & df['rank'].notna()
        removed = int(zero_or_neg.sum())
        if removed:
            logger.warning(f"Replacing {removed} rankings <= 0 with NaN (treated as missing rank).")
        df.loc[zero_or_neg, 'rank'] = pd.NA

    sink = ParquetSink(cfg['ingestion']['parquet_dir'])
    path = sink.write_dataframe('rankings_raw', df, key_cols=['rank_date','weight_class','fighter'], replace=True)
    logger.info(f"Wrote rankings_raw to {path}")


if __name__ == '__main__':
    main()
