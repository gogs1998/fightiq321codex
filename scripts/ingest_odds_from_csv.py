"""
Ingest historical odds from CSV into raw odds (parquet).
Source: FightIQ/data/UFC_betting_odds.csv
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

    csv_path = ROOT.parents[0] / "FightIQ" / "data" / "UFC_betting_odds.csv"
    if not csv_path.exists():
        logger.error(f"Odds CSV not found: {csv_path}")
        sys.exit(2)
    df = pd.read_csv(csv_path)
    # Standardize columns
    rename = {
        'odds_1': 'odds_f1',
        'odds_2': 'odds_f2',
        'fighter_1': 'f1_name',
        'fighter_2': 'f2_name'
    }
    for k,v in rename.items():
        if k in df.columns:
            df = df.rename(columns={k:v})
    if 'fight_url' not in df.columns:
        logger.error("Odds CSV missing fight_url column; cannot proceed.")
        sys.exit(2)

    initial_rows = len(df)
    df = df[df['fight_url'].notna() & (df['fight_url'].astype(str).str.strip() != "")]
    dropped = initial_rows - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} odds rows without fight_url before normalization.")
    # Normalize URLs
    for col in ['fight_url','fighter_1_url','fighter_2_url']:
        if col in df.columns:
            df[col] = df[col].astype('string')
            df[col] = df[col].str.strip().str.rstrip('/')
    if 'fight_url' in df.columns:
        mask_nan = df['fight_url'].str.lower().isin({'nan', 'none', ''})
        removed_extra = int(mask_nan.sum())
        if removed_extra:
            logger.warning(f"Dropping {removed_extra} odds rows with invalid fight_url tokens ('nan'/'none').")
        df = df[~mask_nan]
    # Coerce datetimes
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    df = df.dropna(subset=['fight_url'])

    sink = ParquetSink(cfg['ingestion']['parquet_dir'])
    path = sink.write_dataframe('odds_raw', df, key_cols=['fight_url'], replace=True)
    logger.info(f"Wrote odds_raw to {path}")


if __name__ == '__main__':
    main()
