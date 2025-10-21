from __future__ import annotations

from typing import Iterable
import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_fight_urls_for_event(event_url: str, user_agent: str | None = None, timeout: tuple = (10, 20)) -> pd.DataFrame:
    """
    Given a UFCStats event_url, scrape all fight detail URLs on that card.

    Returns DataFrame with columns: event_url, fight_url
    """
    headers = {"User-Agent": user_agent or "Mozilla/5.0"}
    res = requests.get(event_url, headers=headers, timeout=timeout)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    fights = []
    # Collect all anchors with 'fight-details' path
    for a in soup.select("a"):
        href = a.get("href", "").strip()
        if "fight-details" in href:
            fights.append({"event_url": event_url.rstrip("/"), "fight_url": href.rstrip("/")})
    if not fights:
        return pd.DataFrame(columns=["event_url", "fight_url"])
    df = pd.DataFrame(fights).drop_duplicates(subset=["fight_url"])  # unique fights
    return df


def scrape_fight_urls_for_events(event_urls: Iterable[str], user_agent: str | None = None, timeout: tuple = (10, 20)) -> pd.DataFrame:
    frames = []
    for ev in event_urls:
        try:
            frames.append(scrape_fight_urls_for_event(ev, user_agent=user_agent, timeout=timeout))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["event_url", "fight_url"])
    return pd.concat(frames, axis=0, ignore_index=True).drop_duplicates(subset=["fight_url"]) 

