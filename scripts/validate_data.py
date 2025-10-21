"""
Lightweight data validations for raw/silver/gold layers.
Run this after ingestion/build steps to sanity check schemas and values.

Usage:
  python fightiq_codex/scripts/validate_data.py
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


def _must_exist(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def _check_columns(df: pd.DataFrame, required: list[str], name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"{name} missing columns: {missing}")


def _check_url_series(s: pd.Series, name: str):
    bad = s[s.notna() & ~s.str.startswith('http')]
    if len(bad) > 0:
        raise AssertionError(f"{name}: found {len(bad)} values not starting with http")


def main():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    raw_dir = Path(cfg["paths"]["raw_dir"]) if cfg["ingestion"]["sink"].lower()=="parquet" else None
    if not raw_dir:
        logger.error("Parquet sink is required for this validator")
        sys.exit(2)

    # Raw validations
    events_raw = raw_dir / 'events_raw.parquet'
    fights_raw = raw_dir / 'fights_raw.parquet'
    stats_raw = raw_dir / 'fight_stats_raw.parquet'
    odds_raw = raw_dir / 'odds_raw.parquet'
    ranks_raw = raw_dir / 'rankings_raw.parquet'

    for p in [events_raw, fights_raw, stats_raw]:
        _must_exist(p)

    ev = pd.read_parquet(events_raw)
    _check_columns(ev, ['event_url','event_date','title'], 'events_raw')
    _check_url_series(ev['event_url'].astype(str), 'events_raw.event_url')

    fr = pd.read_parquet(fights_raw)
    _check_columns(fr, ['fight_url','event_url'], 'fights_raw')
    _check_url_series(fr['fight_url'].astype(str), 'fights_raw.fight_url')

    st = pd.read_parquet(stats_raw)
    _check_columns(st, ['fight_url','fighter_name'], 'fight_stats_raw')

    # Optional raw
    if odds_raw.exists():
        od = pd.read_parquet(odds_raw)
        _check_columns(od, ['fight_url','odds_f1','odds_f2'], 'odds_raw')

    if ranks_raw.exists():
        rk = pd.read_parquet(ranks_raw)
        _check_columns(rk, ['rank_date','fighter','weight_class','rank'], 'rankings_raw')

    # Silver validations
    fights_silver = ROOT / 'data' / 'fights_silver.parquet'
    _must_exist(fights_silver)
    fs = pd.read_parquet(fights_silver)
    _check_columns(fs, ['fight_url','event_url','event_date','event_name'], 'fights_silver')

    # Optional silver
    odds_silver = ROOT / 'data' / 'odds_silver.parquet'
    if odds_silver.exists():
        os = pd.read_parquet(odds_silver)
        _check_columns(os, ['fight_url','odds_f1','odds_f2','imp_f1_vigfree','imp_f2_vigfree'], 'odds_silver')

    ranks_silver = ROOT / 'data' / 'rankings_silver.parquet'
    if ranks_silver.exists():
        rs = pd.read_parquet(ranks_silver)
        _check_columns(rs, ['fight_url','event_date','fighter_name','rank'], 'rankings_silver')

    # Gold (if built)
    gold = ROOT / 'data' / 'gold_features.parquet'
    if gold.exists():
        gd = pd.read_parquet(gold)
        if 'fight_url' not in gd.columns:
            raise AssertionError('gold_features missing fight_url')

    logger.info("Validations passed")


if __name__ == '__main__':
    main()

