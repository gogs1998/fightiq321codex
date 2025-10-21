"""
Build a minimal fights_silver view by joining events_raw and fights_raw.
Schema: fight_url, event_url, event_date, event_name
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


def main():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    raw_dir = Path(cfg["paths"]["raw_dir"]) if cfg["ingestion"]["sink"].lower()=="parquet" else None
    if not raw_dir:
        logger.error("This example silver builder supports parquet sink only.")
        sys.exit(2)

    events_path = raw_dir / "events_raw.parquet"
    fights_path = raw_dir / "fights_raw.parquet"
    if not (events_path.exists() and fights_path.exists()):
        logger.error("Missing raw inputs. Run ingest_events.py and ingest_fight_urls.py first.")
        sys.exit(2)

    events = pd.read_parquet(events_path)
    fights = pd.read_parquet(fights_path)

    # Minimal join on event_url
    silver = fights.merge(
        events.rename(columns={"title": "event_name"}), on="event_url", how="left"
    )[["fight_url", "event_url", "event_date", "event_name"]].drop_duplicates()

    out_dir = ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fights_silver.parquet"
    silver.to_parquet(out_path, index=False)
    logger.info(f"Wrote fights_silver to {out_path}")


if __name__ == "__main__":
    main()

