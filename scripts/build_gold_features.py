"""
Build pre-fight gold features with point-in-time (PTI) safeguards.

Inputs (parquet):
- data/fights_silver.parquet: fight_url, event_url, event_date, event_name
- data/rankings_silver.parquet: fight_url, event_date, fighter_name, weight_class, rank
- data/odds_silver.parquet: fight_url, event_date, event_name, odds_f1, odds_f2, imp_f1_vigfree, imp_f2_vigfree
- data/raw/fight_stats_raw.parquet: fight_url, fighter_name, weight_class, totals (sig/total strikes, TD, SUB, REV, CTRL)

Output:
- data/gold_features.parquet: one row per fight with f1/f2 rolling features and matchup deltas; includes odds features and rankings deltas if available.

Notes:
- PTI: rolling features computed from fighter's past fights strictly before current event_date.
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

ROOT = Path(__file__).parents[1]


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    for base in [ROOT, ROOT.parent]:
        candidate = base / path
        if candidate.exists():
            return candidate
    return ROOT / path
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


def _to_seconds(token: str) -> float:
    try:
        token = str(token)
        if ':' in token:
            m, s = token.split(':')
            return float(m) * 60 + float(s)
        return float(token)
    except Exception:
        return 0.0


def build_gold():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    raw_dir = None
    if cfg["ingestion"]["sink"].lower() == "parquet":
        raw_dir = _resolve_path(cfg["paths"]["raw_dir"])
    fights_silver = ROOT / 'data' / 'fights_silver.parquet'
    odds_silver = ROOT / 'data' / 'odds_silver.parquet'
    ranks_silver = ROOT / 'data' / 'rankings_silver.parquet'
    stats_raw = raw_dir / 'fight_stats_raw.parquet' if raw_dir else None

    required = [fights_silver, stats_raw]
    for p in required:
        if not p or not Path(p).exists():
            logger.error(f"Missing input: {p}")
            sys.exit(2)

    fights = pd.read_parquet(fights_silver)
    fights['event_date'] = pd.to_datetime(fights['event_date'], errors='coerce')
    stats = pd.read_parquet(stats_raw)
    stats['fighter_name'] = stats['fighter_name'].astype(str).str.strip()

    # Cast numeric totals
    num_cols = ['sig_strikes_succ','sig_strikes_att','total_strikes_succ','total_strikes_att','takedown_succ','takedown_att','submission_att','reversals']
    for c in num_cols:
        if c in stats.columns:
            stats[c] = pd.to_numeric(stats[c], errors='coerce')
    if 'ctrl_time' in stats.columns:
        stats['ctrl_time_sec'] = stats['ctrl_time'].apply(_to_seconds)

    # Join event_date to stats
    s = stats.merge(fights[['fight_url','event_date']], on='fight_url', how='left')
    s = s.sort_values(['fighter_name','event_date'])

    # Compute rolling features per fighter (PTI) for multiple windows and rates
    base_cols = [c for c in ['sig_strikes_succ','sig_strikes_att','total_strikes_succ','total_strikes_att','takedown_succ','takedown_att','submission_att','reversals','ctrl_time_sec'] if c in s.columns]
    windows = [3,5,10,15,20]
    feats = []
    for name, grp in s.groupby('fighter_name'):
        grp = grp.copy()
        grp[base_cols] = grp[base_cols].fillna(0)
        # derived rates per fight
        grp['sig_acc'] = (grp['sig_strikes_succ'] / grp['sig_strikes_att']).replace([np.inf, -np.inf], 0).fillna(0) if set(['sig_strikes_succ','sig_strikes_att']).issubset(grp.columns) else 0
        grp['td_acc'] = (grp['takedown_succ'] / grp['takedown_att']).replace([np.inf, -np.inf], 0).fillna(0) if set(['takedown_succ','takedown_att']).issubset(grp.columns) else 0
        rate_cols = [c for c in ['sig_acc','td_acc'] if c in grp.columns]
        # recency + experience features
        grp = grp.sort_values('event_date')
        grp['prev_event_date'] = grp['event_date'].shift(1)
        grp['days_since_last_fight'] = (grp['event_date'] - grp['prev_event_date']).dt.days
        grp['days_since_last_fight'] = grp['days_since_last_fight'].fillna(365*5)
        grp['fight_number'] = np.arange(len(grp))
        grp_indexed = grp.set_index('event_date')
        grp_indexed['__ones'] = 1.0
        rolling_365 = grp_indexed['__ones'].rolling('365D').sum().shift(1)
        rolling_730 = grp_indexed['__ones'].rolling('730D').sum().shift(1)
        grp['fights_last_365d'] = rolling_365.reindex(grp_indexed.index).fillna(0).values
        grp['fights_last_730d'] = rolling_730.reindex(grp_indexed.index).fillna(0).values
        grp['avg_days_between_fights_365d'] = grp['days_since_last_fight'].rolling(window=3, min_periods=1).mean().shift(1)
        grp['avg_days_between_fights_365d'] = grp['avg_days_between_fights_365d'].fillna(grp['days_since_last_fight'])

        additional_cols = ['days_since_last_fight','fight_number','fights_last_365d','fights_last_730d','avg_days_between_fights_365d']
        rate_cols.extend(additional_cols)
        all_cols = base_cols + rate_cols
        base_out = grp[['fight_url','fighter_name','event_date']].copy()
        base_out[additional_cols] = grp[additional_cols].values
        frames = [base_out]
        for w in windows:
            meanw = grp[all_cols].rolling(window=w, min_periods=1).mean().shift(1)
            sumw = grp[base_cols].rolling(window=w, min_periods=1).sum().shift(1)
            meanw.columns = [f'{c}_m{w}' for c in all_cols]
            sumw.columns = [f'{c}_s{w}' for c in base_cols]
            frames.extend([meanw, sumw])
        out = pd.concat(frames, axis=1)
        feats.append(out)
    pf = pd.concat(feats, axis=0, ignore_index=True)

    # Create fight-level with f1/f2 sides by joining fighter rows
    # Heuristic: Use two rows per fight_url from stats to identify f1/f2 order by appearance
    order = s[['fight_url','fighter_name','event_date']].drop_duplicates()
    order['idx'] = order.groupby('fight_url').cumcount()
    f1 = order[order['idx']==0].merge(pf, on=['fight_url','fighter_name','event_date'], how='left', suffixes=('',''))
    f2 = order[order['idx']==1].merge(pf, on=['fight_url','fighter_name','event_date'], how='left', suffixes=('',''))
    f1 = f1.add_prefix('f1_')
    f2 = f2.add_prefix('f2_')
    merged = f1.merge(f2, left_on='f1_fight_url', right_on='f2_fight_url', how='inner')
    merged = merged.rename(columns={'f1_fight_url':'fight_url'})
    # unify event_date
    if 'f1_event_date' in merged.columns:
        merged['event_date'] = merged['f1_event_date']

    # Add matchup deltas (f1 - f2) for rolled means across all windows
    for w in windows:
        mean_cols = [c for c in pf.columns if c.endswith(f'_m{w}')]
        for c in mean_cols:
            c1 = f'f1_{c}'
            c2 = f'f2_{c}'
            if c1 in merged.columns and c2 in merged.columns:
                merged[f'delta_{c}'] = merged[c1] - merged[c2]
    # Static matchup deltas
    for col in ['days_since_last_fight','fight_number','fights_last_365d','fights_last_730d','avg_days_between_fights_365d']:
        c1 = f'f1_{col}'
        c2 = f'f2_{col}'
        if c1 in merged.columns and c2 in merged.columns:
            merged[f'delta_{col}'] = merged[c1] - merged[c2]
            merged[f'mean_{col}'] = (merged[c1] + merged[c2]) / 2.0

    # Join odds silver if present
    if odds_silver.exists():
        odds = pd.read_parquet(odds_silver)
        merged = merged.merge(odds[['fight_url','imp_f1_vigfree','imp_f2_vigfree','imp_f1','imp_f2','odds_f1','odds_f2']], on='fight_url', how='left')
        merged['delta_imp_vigfree'] = merged['imp_f1_vigfree'] - merged['imp_f2_vigfree']
        if {'imp_f1','imp_f2'}.issubset(merged.columns):
            merged['delta_imp'] = merged['imp_f1'] - merged['imp_f2']
            merged['mean_imp'] = (merged['imp_f1'] + merged['imp_f2']) / 2.0
        merged['mean_imp_vigfree'] = (merged['imp_f1_vigfree'] + merged['imp_f2_vigfree']) / 2.0
        merged['logit_vigfree'] = np.log(np.clip(merged['imp_f1_vigfree'],1e-6,1)) - np.log(np.clip(merged['imp_f2_vigfree'],1e-6,1))
        merged['odds_delta'] = merged['odds_f1'] - merged['odds_f2']
        merged['vig'] = (1.0 / np.clip(merged['odds_f1'], 1e-6, None) + 1.0 / np.clip(merged['odds_f2'], 1e-6, None)) - 1.0

    # Join rankings silver if present (rank for f1 and f2)
    if ranks_silver.exists():
        rk = pd.read_parquet(ranks_silver)
        r1 = rk.rename(columns={'fighter_name':'f1_fighter_name','rank':'f1_rank'})[['fight_url','f1_fighter_name','f1_rank']]
        r2 = rk.rename(columns={'fighter_name':'f2_fighter_name','rank':'f2_rank'})[['fight_url','f2_fighter_name','f2_rank']]
        merged = merged.merge(r1, on=['fight_url','f1_fighter_name'], how='left')
        merged = merged.merge(r2, on=['fight_url','f2_fighter_name'], how='left')
        merged['delta_rank'] = (merged['f1_rank'].fillna(999) - merged['f2_rank'].fillna(999))
        merged['mean_rank'] = (merged['f1_rank'].fillna(999) + merged['f2_rank'].fillna(999)) / 2.0
        merged['rank_presence'] = ((merged['f1_rank'].notna()).astype(int) + (merged['f2_rank'].notna()).astype(int))

    out_path = ROOT / 'data' / 'gold_features.parquet'
    merged.to_parquet(out_path, index=False)
    logger.info(f"Wrote gold features to {out_path} with {len(merged)} rows")


if __name__ == '__main__':
    build_gold()
