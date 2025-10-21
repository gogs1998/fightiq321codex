FightIQ Merge Plan

Goals
- Unify FightIQ and FightIQ_improved into `fightiq_codex` with one reliable pipeline.
- Preserve leak-safety, validation, and add robust upcoming prediction support.

Steps
1) Core modules (done)
   - Data loader with regex leak filters
   - Temporal splitters and walk-forward
   - Feature-type imputation with indicators
   - Data validator and metrics
   - Calibration and Kelly betting helpers

2) Training + inference (done)
   - Config-driven training with artifact saving
   - Upcoming predictions with feature alignment + calibration + Kelly

3) Backtesting & ROI (next)
   - Add `scripts/backtest_walkforward.py` using WalkForwardSplitter
   - Report accuracy/logloss/ROI over time and stability charts

4) Model selection & tuning (next)
   - Add Optuna search for XGB/LGB
   - Add stacking ensemble path with proper trainers

5) Data ingestion for future cards (optional)
   - Create importers for upcoming cards (CSV/API)
   - Normalize schema to training features

6) CI & tests (next)
   - Unit tests for leakage regex, imputer grouping, calibration
   - Smoke train on small sample

7) Documentation and examples (next)
   - QuickStart with sample CSVs
   - Usage recipes for betting strategies

