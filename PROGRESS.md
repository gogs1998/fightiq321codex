# FightIQ Codex Progress Summary

## Completed to Date

### Data Foundation & Ingestion
- Assembled full leak-safe data layers (raw → silver → gold) across events, fights, statistics, rankings, and odds.
- Added ingestion scripts for each upstream source with PTI guarantees and rolling-window feature engineering.
- Stored historical parquet datasets (`data/`) and upcoming feature matrices (`fightiq_codex/data/`) for reproducibility.

### Modeling & Calibration
- Tuned LightGBM/XGBoost winner models, integrated parity ensemble, and saved guard thresholds for CI.
- Refreshed multi-task method/round baselines (class-weighted LightGBM, early stopping, temperature scaling).
- Logged metrics and artifacts in `PLAN.md`, with regression tests ensuring calibrated probabilities remain normalized.

### Upcoming Predictions
- Built Odds API ingestion (`scripts/ingest_upcoming_from_odds_api.py`) and full PTI feature generator (`scripts/build_upcoming_features.py`) for future fights.
- Hardened `predict_upcoming.py` to read the enriched feature matrix, stack tuned models, and emit parity-aligned forecasts with Kelly sizing.
- Orchestrator now chains ingestion + feature build when `THEODDS_API_KEY` is present; parity, winner, and multi-task model artifacts checked into GitHub.

### Sharing & Documentation
- Added a Kaggle-ready notebook (`fightiq_codex/notebooks/fightiq_codex_pipeline.ipynb`) showing data snapshots, optional pipeline runs, and prediction generation.
- Pushed all core scripts, datasets, and model artifacts to GitHub (`gogs1998/fightiq321codex`) for collaboration.

## Outstanding Work

1. **Calibration & Monitoring**
   - Add per weight class calibration reviews, refresh CI smoke thresholds, and wire in ROI reporting.
2. **Notebook Enhancements**
   - Include exploratory visuals (calibration curves, ROI trends) and narrative cells for Kaggle storytelling.
3. **Automation & Ops**
   - Extend weekly orchestrator with automated validation/prediction reporting; integrate parity alignment assertions.
4. **Documentation & Deployment**
   - Polish README/QuickStart for Kaggle usage, define data publishing workflow (datasets + notebook), and outline agent-driven orchestration.

Artifacts, datasets, and notebook are ready for Kaggle sharing once remaining polish is addressed.***
