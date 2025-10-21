QuickStart

Prereqs
- Python 3.10+
- Install dependencies from the original repo requirements.

Setup
- Ensure training data exists at `data/UFC_full_data_golden.csv` (configurable).
- (Optional) Prepare `data/upcoming_fights.csv` with pre-fight features.

Train
- `python fightiq_codex/scripts/train_baseline.py`
- Artifacts saved to `fightiq_codex/artifacts/<timestamp>/`.

Predict Upcoming
- `python fightiq_codex/scripts/predict_upcoming.py`
- Output CSV in `fightiq_codex/outputs/`.

Walk-Forward Backtest
- `python fightiq_codex/scripts/backtest_walkforward.py`
- Summary CSV in `fightiq_codex/outputs/`.

Config
- Edit `fightiq_codex/config/config.yaml` for paths, models, calibration, and betting settings.

