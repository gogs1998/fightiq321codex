from __future__ import annotations

"""
Lightweight Odds API client (placeholder).
Requires THEODDS_API_KEY in environment or passed explicitly.

Example endpoint (TheOddsAPI):
  https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds?
    apiKey=...&regions=us&markets=h2h&oddsFormat=decimal

We won't call the network here by default; this module provides
fetch functions to be used by an ingest script when credentials exist.
"""

import os
import requests
import pandas as pd
from typing import Iterable, Optional


def fetch_moneyline_odds(api_key: str | None = None, regions: str = "us", markets: str = "h2h") -> pd.DataFrame:
    api_key = api_key or os.getenv("THEODDS_API_KEY")
    if not api_key:
        raise RuntimeError("THEODDS_API_KEY not set")
    url = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    res = requests.get(url, params=params, timeout=(10, 20))
    res.raise_for_status()
    data = res.json()
    rows = []
    for ev in data:
        commence_time = ev.get("commence_time")
        event_title = ev.get("sport_title") or f"{ev.get('home_team', '')} vs {ev.get('away_team', '')}".strip()
        event_id = ev.get("id")
        sport_key = ev.get("sport_key")
        home_team = ev.get("home_team")
        away_team = ev.get("away_team")
        for bk in ev.get("bookmakers", []):
            market = next((m for m in bk.get("markets", []) if m.get("key") == "h2h"), None)
            if not market:
                continue
            outcomes = market.get("outcomes", [])
            if len(outcomes) < 2:
                continue
            o1, o2 = outcomes[0], outcomes[1]
            rows.append(
                {
                    "event_id": event_id,
                    "sport_key": sport_key,
                    "sport_title": ev.get("sport_title"),
                    "event_time": commence_time,
                    "event_name": event_title,
                    "home_team": home_team,
                    "away_team": away_team,
                    "book": bk.get("title"),
                    "book_key": bk.get("key"),
                    "book_last_update": bk.get("last_update"),
                    "market_key": market.get("key"),
                    "outcome_count": len(outcomes),
                    "f1_name": o1.get("name"),
                    "odds_f1": o1.get("price"),
                    "f2_name": o2.get("name"),
                    "odds_f2": o2.get("price"),
                }
            )
    return pd.DataFrame(rows)


def filter_by_event_ids(df: pd.DataFrame, event_ids: Optional[Iterable[str]]) -> pd.DataFrame:
    if not event_ids:
        return df
    event_ids = set(event_ids)
    if not event_ids:
        return df
    return df[df["event_id"].isin(event_ids)].copy()
