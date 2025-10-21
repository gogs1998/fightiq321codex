"""
Ingest per-fight per-fighter totals from UFCStats fight detail pages.
Requires fights_raw.parquet to be present.
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.ingestion.sinks import ParquetSink
from src.ingestion.scrapers.ufcstats_fight_stats import scrape_fight_stats


def main():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    raw_dir = Path(cfg["ingestion"]["parquet_dir"]) if cfg["ingestion"]["sink"].lower()=="parquet" else None
    fights_path = (raw_dir / "fights_raw.parquet") if raw_dir else None
    if not fights_path or not fights_path.exists():
        logger.error("fights_raw.parquet not found. Run ingest_fight_urls.py first.")
        sys.exit(2)

    fights = pd.read_parquet(fights_path)
    fight_urls = fights["fight_url"].dropna().unique().tolist()
    if not fight_urls:
        logger.info("No fight URLs in raw store.")
        return

    headers = cfg["ingestion"].get("user_agent")
    timeouts = (cfg["ingestion"]["timeouts"].get("connect", 10), cfg["ingestion"]["timeouts"].get("read", 20))

    rows = []
    for fu in fight_urls:
        try:
            df = scrape_fight_stats(fu, user_agent=headers, timeout=timeouts)
            if not df.empty:
                rows.append(df)
        except Exception:
            continue

    if not rows:
        logger.info("No fight stats scraped.")
        return
    stats_df = pd.concat(rows, axis=0, ignore_index=True)

    sink = ParquetSink(raw_dir)
    path = sink.write_dataframe("fight_stats_raw", stats_df, key_cols=["fight_url","fighter_name"])
    logger.info(f"Wrote fight_stats_raw to {path}")


if __name__ == "__main__":
    main()

