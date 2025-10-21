from __future__ import annotations

from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd


@dataclass
class EventRecord:
    event_url: str
    event_date: str
    title: str


def scrape_events(limit: Optional[int] = 15, user_agent: Optional[str] = None, timeout: tuple = (10, 20)) -> pd.DataFrame:
    headers = {"User-Agent": user_agent or "Mozilla/5.0"}
    url = "http://ufcstats.com/statistics/events/completed?page=all"
    res = requests.get(url, headers=headers, timeout=timeout)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    rows = [row for row in soup.select("tr.b-statistics__table-row") if row.select_one("a") and row.select_one("span")]

    events: list[EventRecord] = []
    for row in rows:
        a = row.select_one("a")
        span = row.select_one("span")
        href = (a["href"].strip().rstrip("/") if a and a.has_attr("href") else None)
        title = (a.text.strip() if a else None)
        date_txt = (span.text.strip() if span else None)
        if not (href and title and date_txt):
            continue
        try:
            date = pd.to_datetime(date_txt).strftime("%Y-%m-%d")
        except Exception:
            continue
        events.append(EventRecord(event_url=href, event_date=date, title=title))

    if isinstance(limit, int) and limit > 0:
        events = events[:limit]

    df = pd.DataFrame([e.__dict__ for e in events])
    if not df.empty:
        df["event_url"] = df["event_url"].astype(str).str.strip().str.rstrip("/")
        df["title"] = df["title"].astype(str).str.strip()
    return df

