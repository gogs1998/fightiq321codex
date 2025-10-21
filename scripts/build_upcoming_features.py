"""
Construct point-in-time features for upcoming fights so inference uses the full training stack.

Workflow:
1. Load historical fights_silver, fight_stats_raw, optional rankings.
2. Append upcoming fights (from FightIQ/data/upcoming_fights.csv) as synthetic rows.
3. Re-run the rolling aggregations used for gold features to obtain f1/f2 windows + matchup deltas.
4. Join upcoming odds metadata and latest rankings (as-of event date).
5. Persist enriched features to fightiq_codex/data/upcoming_features.{parquet,csv}.

This keeps the inference feature schema aligned with training (same f1_/f2_/delta_* columns).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    for base in [ROOT, ROOT.parent]:
        candidate = base / path
        if candidate.exists():
            return candidate
    return ROOT / path


def _load_base_frames(cfg: Dict) -> Dict[str, pd.DataFrame]:
    fights = pd.read_parquet(ROOT / "data" / "fights_silver.parquet")
    upcoming_path = _resolve_path(cfg["paths"]["upcoming_fights"])
    if not upcoming_path.exists():
        raise FileNotFoundError(f"Upcoming fights CSV missing at {upcoming_path}. Run ingest_upcoming_from_odds_api.py first.")
    upcoming = pd.read_csv(upcoming_path)

    raw_dir = None
    if cfg["ingestion"]["sink"].lower() == "parquet":
        raw_dir = _resolve_path(cfg["paths"]["raw_dir"])
    stats_path = raw_dir / "fight_stats_raw.parquet" if raw_dir else None
    if not stats_path or not stats_path.exists():
        raise FileNotFoundError(f"fight_stats_raw.parquet missing at {stats_path}")
    stats = pd.read_parquet(stats_path)

    rankings_path = ROOT / "data" / "rankings_silver.parquet"
    rankings = pd.read_parquet(rankings_path) if rankings_path.exists() else None

    return {"fights": fights, "stats": stats, "upcoming": upcoming, "rankings": rankings}


def _sanitize_upcoming(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fight_url"] = out["fight_url"].astype(str).str.strip()
    out["event_name"] = out.get("event_name", "").astype(str).str.strip()
    if "event_time_utc" in out.columns:
        out["event_time_utc"] = pd.to_datetime(out["event_time_utc"], utc=True, errors="coerce")
        out["event_time_utc"] = out["event_time_utc"].dt.tz_convert(None)
    else:
        out["event_time_utc"] = pd.NaT
    out["event_date"] = pd.to_datetime(out.get("event_date"), utc=False, errors="coerce")
    missing_dates = out["event_date"].isna()
    if missing_dates.any():
        out.loc[missing_dates, "event_date"] = out.loc[missing_dates, "event_time_utc"]
    out["event_date"] = pd.to_datetime(out["event_date"]).dt.tz_localize(None)
    out["event_time_utc"] = out["event_time_utc"].fillna(out["event_date"])
    out["event_date"] = out["event_date"].fillna(out["event_time_utc"])
    out["event_date"] = pd.to_datetime(out["event_date"])

    rename_map = {
        "fighter1": "f_1_name",
        "fighter2": "f_2_name",
        "fighter1_odds": "f_1_odds",
        "fighter2_odds": "f_2_odds",
        "fighter1_implied_prob": "f_1_implied_prob",
        "fighter2_implied_prob": "f_2_implied_prob",
        "fighter1_prob_vigfree": "f_1_prob_vigfree",
        "fighter2_prob_vigfree": "f_2_prob_vigfree",
    }
    for src, dest in rename_map.items():
        if src in out.columns and dest not in out.columns:
            out[dest] = out[src]
    if {"f_1_name", "f_2_name"}.issubset(out.columns):
        out["f_1_name"] = out["f_1_name"].astype(str).str.strip()
        out["f_2_name"] = out["f_2_name"].astype(str).str.strip()
    else:
        raise ValueError("Upcoming CSV requires f_1_name and f_2_name columns.")
    return out


def _extend_fights(fights: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    template_cols = fights.columns.tolist()
    new_rows = []
    for row in upcoming.itertuples():
        entry = {col: pd.NA for col in template_cols}
        entry["fight_url"] = row.fight_url
        entry["event_url"] = row.fight_url
        entry["event_date"] = row.event_date
        entry["event_name"] = row.event_name
        new_rows.append(entry)
    new_df = pd.DataFrame(new_rows, columns=template_cols)
    combined = pd.concat([fights, new_df], ignore_index=True)
    combined["event_date"] = pd.to_datetime(combined["event_date"], errors="coerce")
    return combined


def _extend_stats(stats: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    stats_cols = stats.columns.tolist()
    new_entries = []
    for row in upcoming.itertuples():
        for fighter_name in (row.f_1_name, row.f_2_name):
            entry = {col: np.nan for col in stats_cols}
            entry["fight_url"] = row.fight_url
            entry["fighter_name"] = fighter_name
            new_entries.append(entry)
    new_df = pd.DataFrame(new_entries, columns=stats_cols)
    return pd.concat([stats, new_df], ignore_index=True)


def _to_seconds(token: str) -> float:
    try:
        token = str(token)
        if ":" in token:
            minutes, seconds = token.split(":")
            return float(minutes) * 60 + float(seconds)
        return float(token)
    except Exception:
        return 0.0


def _compute_rolling_features(stats: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    s = stats.merge(fights[["fight_url", "event_date"]], on="fight_url", how="left")
    s["fighter_name"] = s["fighter_name"].astype(str).str.strip()
    s = s.sort_values(["fighter_name", "event_date"])

    base_cols = [
        col
        for col in [
            "sig_strikes_succ",
            "sig_strikes_att",
            "total_strikes_succ",
            "total_strikes_att",
            "takedown_succ",
            "takedown_att",
            "submission_att",
            "reversals",
            "ctrl_time_sec",
        ]
        if col in s.columns
    ]
    if "ctrl_time" in s.columns and "ctrl_time_sec" not in s.columns:
        s["ctrl_time_sec"] = s["ctrl_time"].apply(_to_seconds)
        base_cols.append("ctrl_time_sec")
    for col in base_cols:
        s[col] = pd.to_numeric(s[col], errors="coerce")

    windows = [3, 5, 10, 15, 20]
    feats = []
    for _, grp in s.groupby("fighter_name"):
        grp = grp.copy()
        grp[base_cols] = grp[base_cols].fillna(0)
        if {"sig_strikes_succ", "sig_strikes_att"}.issubset(grp.columns):
            grp["sig_acc"] = (grp["sig_strikes_succ"] / grp["sig_strikes_att"]).replace([np.inf, -np.inf], 0).fillna(0)
        else:
            grp["sig_acc"] = 0.0
        if {"takedown_succ", "takedown_att"}.issubset(grp.columns):
            grp["td_acc"] = (grp["takedown_succ"] / grp["takedown_att"]).replace([np.inf, -np.inf], 0).fillna(0)
        else:
            grp["td_acc"] = 0.0

        grp = grp.sort_values("event_date")
        grp["prev_event_date"] = grp["event_date"].shift(1)
        grp["days_since_last_fight"] = (grp["event_date"] - grp["prev_event_date"]).dt.days
        grp["days_since_last_fight"] = grp["days_since_last_fight"].fillna(365 * 5)
        grp["fight_number"] = np.arange(len(grp))

        grp_indexed = grp.set_index("event_date")
        grp_indexed["__ones"] = 1.0
        rolling_365 = grp_indexed["__ones"].rolling("365D").sum().shift(1)
        rolling_730 = grp_indexed["__ones"].rolling("730D").sum().shift(1)
        grp["fights_last_365d"] = rolling_365.reindex(grp_indexed.index).fillna(0).values
        grp["fights_last_730d"] = rolling_730.reindex(grp_indexed.index).fillna(0).values
        grp["avg_days_between_fights_365d"] = grp["days_since_last_fight"].rolling(window=3, min_periods=1).mean().shift(1)
        grp["avg_days_between_fights_365d"] = grp["avg_days_between_fights_365d"].fillna(grp["days_since_last_fight"])

        rate_cols = ["sig_acc", "td_acc"]
        additional_cols = [
            "days_since_last_fight",
            "fight_number",
            "fights_last_365d",
            "fights_last_730d",
            "avg_days_between_fights_365d",
        ]
        all_cols = base_cols + rate_cols + additional_cols

        base_out = grp[["fight_url", "fighter_name", "event_date"]].copy()
        base_out[additional_cols] = grp[additional_cols].values
        frames = [base_out]

        for w in windows:
            meanw = grp[all_cols].rolling(window=w, min_periods=1).mean().shift(1)
            sumw = grp[base_cols].rolling(window=w, min_periods=1).sum().shift(1)
            meanw.columns = [f"{col}_m{w}" for col in all_cols]
            sumw.columns = [f"{col}_s{w}" for col in base_cols]
            frames.extend([meanw, sumw])
        out = pd.concat(frames, axis=1)
        feats.append(out)

    return pd.concat(feats, axis=0, ignore_index=True)


def _assemble_fight_level_features(pf: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    order = stats[["fight_url", "fighter_name"]].drop_duplicates()
    order["idx"] = order.groupby("fight_url").cumcount()
    s = stats.merge(order, on=["fight_url", "fighter_name"], how="left")
    s = s.merge(pf, on=["fight_url", "fighter_name"], how="left")

    f1 = s[s["idx"] == 0].copy()
    f2 = s[s["idx"] == 1].copy()
    f1 = f1.add_prefix("f1_")
    f2 = f2.add_prefix("f2_")
    merged = f1.merge(f2, left_on="f1_fight_url", right_on="f2_fight_url", how="inner")
    merged = merged.rename(columns={"f1_fight_url": "fight_url"})
    merged["event_date"] = pd.to_datetime(merged["f1_event_date"])
    return merged


def _add_matchup_deltas(df: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    windows = [3, 5, 10, 15, 20]
    for w in windows:
        mean_cols = [col for col in pf.columns if col.endswith(f"_m{w}")]
        for col in mean_cols:
            c1 = f"f1_{col}"
            c2 = f"f2_{col}"
            if c1 in df.columns and c2 in df.columns:
                df[f"delta_{col}"] = df[c1] - df[c2]
    for col in [
        "days_since_last_fight",
        "fight_number",
        "fights_last_365d",
        "fights_last_730d",
        "avg_days_between_fights_365d",
    ]:
        c1 = f"f1_{col}"
        c2 = f"f2_{col}"
        if c1 in df.columns and c2 in df.columns:
            df[f"delta_{col}"] = df[c1] - df[c2]
            df[f"mean_{col}"] = (df[c1] + df[c2]) / 2.0
    return df


def _merge_rankings(upcoming: pd.DataFrame, rankings: pd.DataFrame | None) -> pd.DataFrame:
    if rankings is None or rankings.empty:
        upcoming["f1_rank"] = np.nan
        upcoming["f2_rank"] = np.nan
        upcoming["delta_rank"] = np.nan
        upcoming["mean_rank"] = np.nan
        upcoming["rank_presence"] = 0
        return upcoming

    ranks = rankings.copy()
    ranks["event_date"] = pd.to_datetime(ranks["event_date"], errors="coerce")
    ranks = ranks.dropna(subset=["event_date"])
    grouped = {
        fighter: grp.sort_values("event_date").reset_index(drop=True)
        for fighter, grp in ranks.groupby("fighter_name")
    }

    def _latest_rank(fighter: str, event_date: pd.Timestamp) -> float:
        grp = grouped.get(fighter)
        if grp is None or grp.empty or pd.isna(event_date):
            return np.nan
        event_dt = pd.Timestamp(event_date)
        dates = grp["event_date"].to_numpy(dtype="datetime64[ns]")
        idx = dates.searchsorted(event_dt.to_datetime64(), side="right") - 1
        if idx < 0:
            return np.nan
        return float(grp.iloc[idx]["rank"]) if pd.notna(grp.iloc[idx]["rank"]) else np.nan

    upcoming["f1_rank"] = [
        _latest_rank(fighter, event_dt) for fighter, event_dt in zip(upcoming["f_1_name"], upcoming["event_date"])
    ]
    upcoming["f2_rank"] = [
        _latest_rank(fighter, event_dt) for fighter, event_dt in zip(upcoming["f_2_name"], upcoming["event_date"])
    ]
    upcoming["delta_rank"] = (upcoming["f1_rank"].fillna(999) - upcoming["f2_rank"].fillna(999))
    upcoming["mean_rank"] = (upcoming["f1_rank"].fillna(999) + upcoming["f2_rank"].fillna(999)) / 2.0
    upcoming["rank_presence"] = upcoming["f1_rank"].notna().astype(int) + upcoming["f2_rank"].notna().astype(int)
    return upcoming


def build_upcoming_features() -> None:
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    frames = _load_base_frames(cfg)
    fights = frames["fights"]
    stats = frames["stats"]
    upcoming = _sanitize_upcoming(frames["upcoming"])
    rankings = frames["rankings"]

    if upcoming.empty:
        logger.warning("Upcoming fights CSV is empty; nothing to compute.")
        return

    # Extend base tables with synthetic upcoming fights
    fights_ext = _extend_fights(fights, upcoming)
    stats_ext = _extend_stats(stats, upcoming)

    stats_ext["fighter_name"] = stats_ext["fighter_name"].astype(str).str.strip()
    pf = _compute_rolling_features(stats_ext, fights_ext)
    fight_level = _assemble_fight_level_features(pf, stats_ext)
    fight_level = _add_matchup_deltas(fight_level, pf)

    # Keep only the upcoming fights
    fight_level = fight_level[fight_level["fight_url"].isin(upcoming["fight_url"])].copy()
    if fight_level.empty:
        logger.error("Failed to derive features for upcoming fights; verify fighter name matching.")
        sys.exit(3)

    # Merge odds + metadata
    merged = upcoming.merge(fight_level, on="fight_url", how="left", suffixes=("", "_hist"))
    merged = _merge_rankings(merged, rankings)

    # Odds-derived features mirroring gold builder
    merged["imp_f1_vigfree"] = merged["f_1_prob_vigfree"]
    merged["imp_f2_vigfree"] = merged["f_2_prob_vigfree"]
    merged["imp_f1"] = merged["f_1_implied_prob"]
    merged["imp_f2"] = merged["f_2_implied_prob"]
    merged["odds_f1"] = merged["f_1_odds"]
    merged["odds_f2"] = merged["f_2_odds"]

    merged["delta_imp_vigfree"] = merged["imp_f1_vigfree"] - merged["imp_f2_vigfree"]
    merged["mean_imp_vigfree"] = (merged["imp_f1_vigfree"] + merged["imp_f2_vigfree"]) / 2.0
    if {"imp_f1", "imp_f2"}.issubset(merged.columns):
        merged["delta_imp"] = merged["imp_f1"] - merged["imp_f2"]
        merged["mean_imp"] = (merged["imp_f1"] + merged["imp_f2"]) / 2.0
    merged["logit_vigfree"] = np.log(np.clip(merged["imp_f1_vigfree"], 1e-6, 1)) - np.log(
        np.clip(merged["imp_f2_vigfree"], 1e-6, 1)
    )
    merged["odds_delta"] = merged["odds_f1"] - merged["odds_f2"]
    merged["vig"] = (1.0 / np.clip(merged["odds_f1"], 1e-6, None) + 1.0 / np.clip(merged["odds_f2"], 1e-6, None)) - 1.0

    # Ensure fighter metadata align with training conventions
    merged["f1_fighter_name"] = merged.get("f1_fighter_name", merged["f_1_name"])
    merged["f2_fighter_name"] = merged.get("f2_fighter_name", merged["f_2_name"])
    merged["event_date"] = pd.to_datetime(merged["event_date"])

    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0.0)

    output_dir = ROOT / "fightiq_codex" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "upcoming_features.parquet"
    csv_path = output_dir / "upcoming_features.csv"
    merged.to_parquet(parquet_path, index=False)
    merged.to_csv(csv_path, index=False)
    logger.success("Wrote upcoming feature matrix to {} (rows={}, cols={})", parquet_path, len(merged), merged.shape[1])


if __name__ == "__main__":
    build_upcoming_features()
