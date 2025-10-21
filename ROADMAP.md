Accuracy and ROI Roadmap

Short-Term (1–2 weeks)
- Probability calibration: keep Platt/Isotonic on validation; verify ECE/Brier.
- Feature pruning: remove low-importance/noisy features (L1 for LR; SHAP for tree models).
- Odds-agnostic vs odds-aware: maintain both tracks; report deltas.
- Walk-forward backtesting: establish rolling metrics and ROI consistency.
- Betting policy: cap Kelly, enforce min edge, and diversify across fights.

Medium-Term (3–6 weeks)
- Hyperparameter search (Optuna): 100–300 trials for XGB/LGB with time-series CV.
- Stacking ensemble: XGB + LGB + calibrated LR meta-learner using OOF predictions.
- Class-conditional calibration: per-weight-class calibration if sufficient data.
- Robustness checks: sensitivity by recency windows and opponent-style features.
- Data augmentation: engineered matchup deltas and interaction terms.

Long-Term (6–12 weeks)
- Adversarial validation to detect train/val shift.
- Online calibration drift monitoring on latest events.
- Feature store with versioning and reproducible point-in-time joins.
- Live odds integration with sanity checks and latency-aware ingestion.
- Multi-objective optimization: accuracy, logloss, ROI, drawdown.

Future Predictions Stability
- Strict align-to-training columns; any missing -> imputer handles.
- Validate schema before inference; fail-fast if critical fields absent (fighter IDs, names).
- Fallbacks when odds missing: predict probs only; bet sizes omitted.
- Use latest trained artifacts automatically; pin via run directory.

