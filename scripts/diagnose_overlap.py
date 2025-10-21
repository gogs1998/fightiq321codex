"""
Diagnose overlap between gold features and original golden CSV labels.
Reports counts by year and overall coverage.
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


def main():
    gold_path = ROOT / 'data' / 'gold_features.parquet'
    lab_path = ROOT.parents[0] / 'FightIQ' / 'data' / 'UFC_full_data_golden.csv'
    if not gold_path.exists() or not lab_path.exists():
        print("Missing gold features or labels; build gold and ensure labels CSV present.")
        return
    gold = pd.read_parquet(gold_path)
    lab = pd.read_csv(lab_path)
    for df in (gold, lab):
        df['fight_url'] = df['fight_url'].astype(str).str.strip().str.rstrip('/')
    if 'event_date' in gold.columns:
        gold['event_date'] = pd.to_datetime(gold['event_date'], errors='coerce')
        gold['year'] = gold['event_date'].dt.year
    else:
        gold['year'] = None
    if 'event_date' in lab.columns:
        lab['event_date'] = pd.to_datetime(lab['event_date'], errors='coerce')
        lab['year'] = lab['event_date'].dt.year
    else:
        lab['year'] = None

    merged = gold[['fight_url','year']].merge(lab[['fight_url','year']].rename(columns={'year':'label_year'}), on='fight_url', how='left')
    coverage = (~merged['label_year'].isna()).mean()
    print(f"Gold fights: {len(gold):,}; Labels attached: {merged['label_year'].notna().sum():,} ({coverage:.1%})")

    if 'year' in merged.columns:
        by_year = merged.groupby('year')['label_year'].apply(lambda s: (~s.isna()).mean()).reset_index(name='coverage')
        print("\nCoverage by year (gold year):")
        print(by_year.to_string(index=False))

    # Winner label completeness
    if 'winner_encoded' in lab.columns:
        lab_nonnull = lab[['fight_url','winner_encoded']].dropna().shape[0]
        print(f"Label rows with winner_encoded: {lab_nonnull:,}")


if __name__ == '__main__':
    main()

