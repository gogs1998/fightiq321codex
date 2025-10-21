"""
Build odds_silver by joining fights_silver and odds_raw on fight_url.
Adds implied probabilities and vig-removed (normalized) implieds.
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
    fights_silver_path = ROOT / "data" / "fights_silver.parquet"
    if not raw_dir or not (raw_dir / 'odds_raw.parquet').exists() or not fights_silver_path.exists():
        logger.error("Missing inputs. Run ingest and build_silver_fights first.")
        sys.exit(2)

    odds = pd.read_parquet(raw_dir / 'odds_raw.parquet')
    fights = pd.read_parquet(fights_silver_path)

    # Minimal columns
    for col in ['fight_url','odds_f1','odds_f2']:
        if col not in odds.columns:
            logger.error(f"odds_raw missing column: {col}")
            sys.exit(2)
    o = odds[['fight_url','odds_f1','odds_f2']].drop_duplicates(subset=['fight_url'])
    o['odds_f1'] = pd.to_numeric(o['odds_f1'], errors='coerce')
    o['odds_f2'] = pd.to_numeric(o['odds_f2'], errors='coerce')

    silver = fights.merge(o, on='fight_url', how='left')
    # Implied probabilities
    silver['imp_f1'] = 1.0 / silver['odds_f1']
    silver['imp_f2'] = 1.0 / silver['odds_f2']
    s = silver['imp_f1'] + silver['imp_f2']
    silver['imp_f1_vigfree'] = silver['imp_f1'] / s
    silver['imp_f2_vigfree'] = silver['imp_f2'] / s

    out_path = ROOT / 'data' / 'odds_silver.parquet'
    silver.to_parquet(out_path, index=False)
    logger.info(f"Wrote odds_silver to {out_path}")


if __name__ == '__main__':
    main()

