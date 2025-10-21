from __future__ import annotations

import requests
from bs4 import BeautifulSoup
import pandas as pd


def _extract_stats_list(soup: BeautifulSoup):
    # Returns list of p tags containing stats in fight details page
    return soup.select('p.b-fight-details__table-text')


def _safe_text(el):
    return el.text.strip() if el else ""


def _parse_counts(token: str):
    # format like '46 of 120' -> (succ, att)
    try:
        left, right = token.split(' of ')
        return left.strip(), right.strip()
    except Exception:
        return "", ""


def scrape_fight_stats(fight_url: str, user_agent: str | None = None, timeout: tuple = (10, 20)) -> pd.DataFrame:
    headers = {"User-Agent": user_agent or "Mozilla/5.0"}
    res = requests.get(fight_url, headers=headers, timeout=timeout)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    stats = _extract_stats_list(soup)
    # attempt to parse weight class from page title area
    weight_class = ""
    ttl = soup.select_one('i.b-fight-details__fight-title')
    if ttl and ttl.text:
        # often contains 'Lightweight Bout' or similar
        t = ttl.text.strip()
        if 'Bout' in t:
            weight_class = t.replace('Bout','').strip()

    # fighter names
    names = [a.text.strip() for a in soup.select('a.b-fight-details__person-link')][:2]
    f1_name = names[0] if len(names) > 0 else ""
    f2_name = names[1] if len(names) > 1 else ""

    # Attempt to parse totals per fighter using known order (fragile but common)
    def fighter_row(i: int):
        # i=0 for fighter1, i=1 for fighter2
        kd = _safe_text(stats[2 + i]) if len(stats) > 2 + i else ""
        sig_succ, sig_att = _parse_counts(_safe_text(stats[4 + i]) if len(stats) > 4 + i else "")
        tot_succ, tot_att = _parse_counts(_safe_text(stats[8 + i]) if len(stats) > 8 + i else "")
        td_succ, td_att = _parse_counts(_safe_text(stats[10 + i]) if len(stats) > 10 + i else "")
        sub_att = _safe_text(stats[14 + i]) if len(stats) > 14 + i else ""
        rev = _safe_text(stats[16 + i]) if len(stats) > 16 + i else ""
        ctrl = _safe_text(stats[18 + i]) if len(stats) > 18 + i else ""
        return {
            "fighter_name": f1_name if i == 0 else f2_name,
            "weight_class": weight_class,
            "knockdowns": kd,
            "sig_strikes_succ": sig_succ,
            "sig_strikes_att": sig_att,
            "total_strikes_succ": tot_succ,
            "total_strikes_att": tot_att,
            "takedown_succ": td_succ,
            "takedown_att": td_att,
            "submission_att": sub_att,
            "reversals": rev,
            "ctrl_time": ctrl,
        }

    rows = []
    rows.append(fighter_row(0))
    rows.append(fighter_row(1))
    for r in rows:
        r["fight_url"] = fight_url.rstrip("/")
    return pd.DataFrame(rows)
