"""
Ingest UFCStats completed events page into the raw store (parquet by default).

Usage:
  python fightiq_codex/scripts/ingest_events.py [--limit 15|ALL]
"""

import sys
from pathlib import Path
import argparse
from loguru import logger
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.ingestion.scrapers.ufcstats_events import scrape_events
from src.ingestion.sinks import ParquetSink, BigQuerySink


def main():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", default="15", help="Number of events or 'ALL'")
    args = ap.parse_args()

    limit = args.limit
    if isinstance(limit, str) and limit.upper() != "ALL":
        try:
            limit = int(limit)
        except Exception:
            logger.error("--limit must be integer or 'ALL'")
            sys.exit(2)
    elif isinstance(limit, str) and limit.upper() == "ALL":
        limit = None

    headers = cfg["ingestion"].get("user_agent")
    timeouts = (cfg["ingestion"]["timeouts"].get("connect", 10), cfg["ingestion"]["timeouts"].get("read", 20))

    logger.info("Scraping UFCStats completed events...")
    df = scrape_events(limit=limit, user_agent=headers, timeout=timeouts)
    if df.empty:
        logger.info("No events scraped.")
        return

    sink_type = cfg["ingestion"].get("sink", "parquet").lower()
    if sink_type == "parquet":
        sink = ParquetSink(cfg["ingestion"]["parquet_dir"])
        path = sink.write_dataframe("events_raw", df, key_cols=["event_url"])
        logger.info(f"Wrote events to {path}")
    elif sink_type == "bigquery":
        proj = cfg["bigquery"]["project_id"]
        dset = cfg["bigquery"]["dataset"]
        sink = BigQuerySink(proj, dset)
        sink.write_dataframe("events_raw", df, key_cols=["event_url"])
        logger.info("Inserted events to BigQuery" )
    else:
        logger.error(f"Unknown sink: {sink_type}")
        sys.exit(2)


if __name__ == "__main__":
    main()

