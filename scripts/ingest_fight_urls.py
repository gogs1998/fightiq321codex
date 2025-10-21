"""
Ingest fight URLs from UFCStats event pages into raw store.
Requires events_raw.parquet to be present (from ingest_events.py).
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.ingestion.sinks import ParquetSink
from src.ingestion.scrapers.ufcstats_fight_urls import scrape_fight_urls_for_events


def main():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    raw_dir = Path(cfg["ingestion"]["parquet_dir"]) if cfg["ingestion"]["sink"].lower()=="parquet" else None
    events_path = (raw_dir / "events_raw.parquet") if raw_dir else None
    if not events_path or not events_path.exists():
        logger.error("events_raw.parquet not found. Run ingest_events.py first.")
        sys.exit(2)

    events = pd.read_parquet(events_path)
    if events.empty:
        logger.info("No events in raw store.")
        return
    event_urls = events["event_url"].dropna().unique().tolist()

    logger.info(f"Scraping fight URLs for {len(event_urls)} events...")
    headers = cfg["ingestion"].get("user_agent")
    timeouts = (cfg["ingestion"]["timeouts"].get("connect", 10), cfg["ingestion"]["timeouts"].get("read", 20))
    fights_df = scrape_fight_urls_for_events(event_urls, user_agent=headers, timeout=timeouts)
    if fights_df.empty:
        logger.info("No fight URLs found.")
        return

    sink = ParquetSink(raw_dir)
    path = sink.write_dataframe("fights_raw", fights_df, key_cols=["fight_url"])
    logger.info(f"Wrote fights_raw to {path}")


if __name__ == "__main__":
    main()

