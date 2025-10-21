FightIQ Codex – Unified, Leak‑Safe UFC Prediction Pipeline

Overview
- Unified best of both FightIQ repos into a single, clean, configurable pipeline.
- Strict leakage prevention, time‑series validation, calibrated probabilities, and betting strategy support.
- Robust “upcoming fights” inference with feature alignment and fail‑safe preprocessing.

Quick Start
- Configure: edit `fightiq_codex/config/config.yaml`.
- Train: `python fightiq_codex/scripts/train_baseline.py`
- Predict upcoming: `python fightiq_codex/scripts/predict_upcoming.py`

Key Features
- Regex‑based leak filters + temporal split validation.
- Feature‑type imputation (rolling=0, others=median) with optional missingness indicators.
- Optional calibrated outputs (Platt or Isotonic) for better ROI decisions.
- Ensemble stacking (OOF) ready for advanced use.
- Kelly Criterion sizing with caps for risk control.

Structure
- `config/` – YAML config for paths, splits, models, options.
- `src/data/` – loaders, splitters, preprocessing, validation.
- `src/evaluation/` – metrics.
- `src/models/` – ensemble + calibration.
- `src/betting/` – Kelly.
- `scripts/` – train/predict entry points.
- `tests/` – leakage checks.
- `artifacts/` – saved models, imputers, scalers, features (created on train).
- `outputs/` – predictions and reports.

Notes
- No external downloads; upcoming predictions expect a CSV per config.
- Unicode/emojis avoided to keep logs portable across terminals.

