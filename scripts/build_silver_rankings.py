"""
Build rankings_silver by assigning per-fighter rank as of event_date using rankings_raw.
Joins fighters (by name) from fight_stats_raw to event dates from fights_silver.
Outputs: fight_url, event_date, fighter_name, weight_class, rank (as of event_date)
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
    stats_raw_path = raw_dir / 'fight_stats_raw.parquet' if raw_dir else None
    ranks_raw_path = raw_dir / 'rankings_raw.parquet' if raw_dir else None
    if not all([raw_dir, fights_silver_path.exists(), stats_raw_path.exists(), ranks_raw_path.exists()]):
        logger.error("Missing inputs. Need fights_silver, fight_stats_raw, rankings_raw.")
        sys.exit(2)

    fights = pd.read_parquet(fights_silver_path)
    stats = pd.read_parquet(stats_raw_path)
    ranks = pd.read_parquet(ranks_raw_path)

    # Normalize
    stats['fighter_name'] = stats['fighter_name'].astype(str).str.strip()
    ranks['fighter'] = ranks['fighter'].astype(str).str.strip()
    # Merge event date onto stats
    df = stats.merge(fights[['fight_url','event_date']], on='fight_url', how='left')
    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    ranks['rank_date'] = pd.to_datetime(ranks['rank_date'], errors='coerce')

    # For now, match by fighter name only (weight class optional)
    # Compute last known rank on/before event_date per fighter
    ranks = ranks.sort_values(['fighter','rank_date']).reset_index(drop=True)
    # Build mapping per fighter of rank timeline
    def get_rank_asof(name: str, dt: pd.Timestamp):
        sub = ranks[ranks['fighter'] == name]
        if sub.empty:
            return None
        sub2 = sub[sub['rank_date'] <= dt]
        if sub2.empty:
            return None
        return int(sub2.iloc[-1]['rank']) if 'rank' in sub2.columns and pd.notna(sub2.iloc[-1]['rank']) else None

    out_rows = []
    for row in df.itertuples(index=False):
        rk = get_rank_asof(row.fighter_name, row.event_date)
        out_rows.append({
            'fight_url': row.fight_url,
            'event_date': row.event_date,
            'fighter_name': row.fighter_name,
            'weight_class': getattr(row, 'weight_class', None),
            'rank': rk
        })
    out = pd.DataFrame(out_rows)
    out_path = ROOT / 'data' / 'rankings_silver.parquet'
    out.to_parquet(out_path, index=False)
    logger.info(f"Wrote rankings_silver to {out_path}")


if __name__ == '__main__':
    main()

