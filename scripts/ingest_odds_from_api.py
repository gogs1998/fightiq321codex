"""
Fetch current moneyline odds via an external API and write to odds_raw (parquet).
Requires THEODDS_API_KEY env var.
"""

import os
import sys
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.ingestion.sinks import ParquetSink
from src.ingestion.scrapers.odds_api import fetch_moneyline_odds


def main():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    api_key = os.getenv("THEODDS_API_KEY")
    if not api_key:
        logger.error("THEODDS_API_KEY not set in environment")
        sys.exit(2)

    df = fetch_moneyline_odds(api_key=api_key)
    if df.empty:
        logger.info("No odds rows fetched.")
        return

    sink = ParquetSink(cfg['ingestion']['parquet_dir'])
    path = sink.write_dataframe('odds_raw', df, key_cols=['event_name','book','f1_name','f2_name'])
    logger.info(f"Wrote odds_raw to {path}")


if __name__ == '__main__':
    main()

