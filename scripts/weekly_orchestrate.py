"""
Weekly orchestration stub (scrape → ingest → build → validate → train), Agent Kit ready.
This script sequences the core steps with logging; replace with Agent Kit flows later.
"""

import os
import sys
from pathlib import Path
from loguru import logger
import subprocess

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


def run(cmd: list[str]):
    logger.info(f"RUN: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(ROOT.parents[0]), capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(res.stdout)
        logger.error(res.stderr)
        raise SystemExit(res.returncode)
    else:
        logger.info(res.stdout.strip())


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    # Ingest (UFCStats + CSV fallbacks)
    run([sys.executable, str(ROOT / 'scripts' / 'ingest_events.py'), '--limit', '100'])
    run([sys.executable, str(ROOT / 'scripts' / 'ingest_fight_urls.py')])
    run([sys.executable, str(ROOT / 'scripts' / 'ingest_fight_stats.py')])
    # Historical odds & rankings from CSV (replace with API scrapes as desired)
    run([sys.executable, str(ROOT / 'scripts' / 'ingest_odds_from_csv.py')])
    run([sys.executable, str(ROOT / 'scripts' / 'ingest_rankings_from_csv.py')])
    # Build silver + gold
    run([sys.executable, str(ROOT / 'scripts' / 'build_silver_fights.py')])
    run([sys.executable, str(ROOT / 'scripts' / 'build_silver_odds.py')])
    run([sys.executable, str(ROOT / 'scripts' / 'build_silver_rankings.py')])
    run([sys.executable, str(ROOT / 'scripts' / 'build_gold_features.py')])
    if os.getenv("THEODDS_API_KEY"):
        run([sys.executable, str(ROOT / 'scripts' / 'ingest_upcoming_from_odds_api.py')])
        run([sys.executable, str(ROOT / 'scripts' / 'build_upcoming_features.py')])
    else:
        logger.warning("THEODDS_API_KEY not set; skipping upcoming odds + feature ingestion.")
    # Validate
    run([sys.executable, str(ROOT / 'scripts' / 'validate_data.py')])
    run([sys.executable, str(ROOT / 'scripts' / 'run_ge_validations.py')])
    # Train models
    run([sys.executable, str(ROOT / 'scripts' / 'train_multitask.py')])
    run([sys.executable, str(ROOT / 'scripts' / 'train_winner_enhanced.py'), '--min-val-acc', '0.71', '--min-test-acc', '0.62'])
    run([sys.executable, str(ROOT / 'scripts' / 'train_winner_parity.py')])
    logger.info("Weekly orchestration complete.")


if __name__ == '__main__':
    main()
