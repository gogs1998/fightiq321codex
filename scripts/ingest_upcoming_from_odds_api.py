"""
Fetch upcoming UFC fights via the Odds API and materialise an inference-ready CSV.

Creates/overwrites FightIQ/data/upcoming_fights.csv (config-driven) with columns:
- fight_url (synthetic, unique per matchup)
- event_id / event_name / event_time_utc / event_date
- f_1_name / f_2_name
- f_1_odds / f_2_odds (consensus decimal odds averaged across books)
- f_1_odds_min / f_1_odds_max / f_2_odds_min / f_2_odds_max
- implied probabilities (raw + vig-removed)
- metadata about contributing books

Requires an Odds API key either via --api-key or THEODDS_API_KEY env var.
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.ingestion.scrapers.odds_api import fetch_moneyline_odds  # noqa: E402


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    for base in [ROOT, ROOT.parent]:
        candidate = base / path
        if candidate.exists():
            return candidate
    return ROOT / path


def _slugify(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "unknown"


def _build_fight_url(event_time: datetime, f1: str, f2: str) -> str:
    iso_date = event_time.strftime("%Y%m%dT%H%MZ")
    return f"https://odds-api.com/fight/{iso_date}-{_slugify(f1)}-vs-{_slugify(f2)}"


def _normalise_names(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["f1_name", "f2_name", "event_name", "book"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def _to_datetime_utc(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt.dt.tz_convert(timezone.utc) if hasattr(dt.dt, "tz_convert") else dt


def _aggregate_fight(group: pd.DataFrame) -> pd.Series:
    base_row = group.iloc[0]
    f1 = base_row["f1_name"]
    f2 = base_row["f2_name"]
    event_time = base_row["event_time"]

    f1_odds: list[float] = []
    f2_odds: list[float] = []
    books = []
    last_updates = []

    for _, row in group.iterrows():
        name_a = row["f1_name"]
        name_b = row["f2_name"]
        odd_a = row["odds_f1"]
        odd_b = row["odds_f2"]

        if pd.isna(odd_a) or pd.isna(odd_b):
            continue

        if name_a == f1 and name_b == f2:
            f1_odds.append(float(odd_a))
            f2_odds.append(float(odd_b))
        elif name_a == f2 and name_b == f1:
            f1_odds.append(float(odd_b))
            f2_odds.append(float(odd_a))
        else:
            # Unknown ordering, fall back to mapping by fighter names where possible.
            mapping = {
                name_a: float(odd_a),
                name_b: float(odd_b),
            }
            if f1 in mapping and f2 in mapping:
                f1_odds.append(mapping[f1])
                f2_odds.append(mapping[f2])
            else:
                continue
        books.append(row.get("book"))
        last_updates.append(row.get("book_last_update"))

    f1_odds = np.array(f1_odds, dtype=float)
    f2_odds = np.array(f2_odds, dtype=float)
    if f1_odds.size == 0 or f2_odds.size == 0:
        return pd.Series(dtype=object)

    f1_mean = float(np.nanmean(f1_odds))
    f2_mean = float(np.nanmean(f2_odds))
    f1_min = float(np.nanmin(f1_odds))
    f2_min = float(np.nanmin(f2_odds))
    f1_max = float(np.nanmax(f1_odds))
    f2_max = float(np.nanmax(f2_odds))

    f1_implied = 1.0 / f1_mean if f1_mean > 0 else math.nan
    f2_implied = 1.0 / f2_mean if f2_mean > 0 else math.nan
    implied_sum = f1_implied + f2_implied if all(not math.isnan(x) for x in [f1_implied, f2_implied]) else math.nan
    if implied_sum and not math.isnan(implied_sum):
        f1_prob = f1_implied / implied_sum
        f2_prob = f2_implied / implied_sum
    else:
        f1_prob = math.nan
        f2_prob = math.nan

    return pd.Series(
        {
            "fight_url": _build_fight_url(event_time, f1, f2),
            "event_id": base_row.get("event_id"),
            "sport_key": base_row.get("sport_key"),
            "sport_title": base_row.get("sport_title"),
            "event_name": base_row.get("event_name"),
            "event_time_utc": event_time,
            "event_date": event_time.date(),
            "f_1_name": f1,
            "f_2_name": f2,
            "f_1_odds": round(f1_mean, 6),
            "f_2_odds": round(f2_mean, 6),
            "f_1_odds_min": round(f1_min, 6),
            "f_2_odds_min": round(f2_min, 6),
            "f_1_odds_max": round(f1_max, 6),
            "f_2_odds_max": round(f2_max, 6),
            "f_1_implied_prob": f1_implied,
            "f_2_implied_prob": f2_implied,
            "f_1_prob_vigfree": f1_prob,
            "f_2_prob_vigfree": f2_prob,
            "vig_percent": (implied_sum - 1.0) * 100 if implied_sum else math.nan,
            "num_books": int(len(set(filter(None, books)))),
            "books": ",".join(sorted(set(filter(None, books)))),
            "book_last_update": max(filter(None, last_updates), default=None),
        }
    )


def ingest_upcoming(api_key: str | None, output_path: Path, regions: str, markets: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Fetching upcoming odds from TheOddsAPI (regions={}, markets={})...", regions, markets)
    raw_df = fetch_moneyline_odds(api_key=api_key, regions=regions, markets=markets)
    if raw_df.empty:
        logger.warning("No rows returned from Odds API.")
        return raw_df, raw_df

    raw_df = _normalise_names(raw_df)
    raw_df["event_time"] = _to_datetime_utc(raw_df["event_time"])
    raw_df = raw_df.dropna(subset=["event_time", "f1_name", "f2_name"])

    raw_df["fight_group"] = raw_df.apply(
        lambda r: f"{r.get('event_id') or ''}::{_slugify(r['f1_name'])}-vs-{_slugify(r['f2_name'])}::{r['event_time'].strftime('%Y%m%dT%H%M')}",
        axis=1,
    )

    agg_df = (
        raw_df.groupby("fight_group", dropna=False)
        .apply(_aggregate_fight, include_groups=False)
        .dropna(subset=["f_1_odds", "f_2_odds"])
    )
    agg_df = agg_df.reset_index(drop=True)
    agg_df = agg_df.sort_values(["event_date", "event_time_utc", "fight_url"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(output_path, index=False)
    logger.info("Wrote {} upcoming fights to {}", len(agg_df), output_path)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = ROOT / "outputs" / "upcoming_fetch"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"upcoming_odds_raw_{ts}.csv"
    raw_df.to_csv(snapshot_path, index=False)
    logger.info("Saved raw odds snapshot ({} rows) to {}", len(raw_df), snapshot_path)
    return raw_df, agg_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch upcoming UFC fights from Odds API and materialise inference CSV.")
    parser.add_argument("--api-key", type=str, default=None, help="Odds API key (falls back to THEODDS_API_KEY).")
    parser.add_argument("--regions", type=str, default="us,uk,eu", help="Comma separated odds regions (default: us,uk,eu).")
    parser.add_argument("--markets", type=str, default="h2h", help="Odds markets to request (default: h2h).")
    parser.add_argument("--output", type=str, default=None, help="Optional override for upcoming_fights.csv output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(ROOT / "config" / "config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    api_key = args.api_key or os.getenv("THEODDS_API_KEY")
    if not api_key:
        logger.error("Provide Odds API key via --api-key or THEODDS_API_KEY env var.")
        sys.exit(2)

    output_path = _resolve_path(args.output or cfg["paths"]["upcoming_fights"])
    raw_df, agg_df = ingest_upcoming(api_key, output_path, args.regions, args.markets)
    if agg_df.empty:
        logger.warning("No aggregated fights produced; upstream CSV unchanged.")
    else:
        logger.success("Upcoming ingestion complete: {} fights ready for prediction.", len(agg_df))


if __name__ == "__main__":
    main()
