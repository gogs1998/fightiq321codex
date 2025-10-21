# FightIQ Codex – End‑to‑End Agent Pipeline Plan

This plan turns FightIQ Codex into a fully agent‑operated pipeline that scrapes, ingests, validates, engineers features, trains/calibrates multi‑task models (winner/method/round), evaluates ROI, and deploys weekly predictions with risk‑controlled betting plans.
Reference repos/dirs to reuse:
- Historical data and notebooks (Kaggle source lineage): `FightIQ/` (data/, notebooks) and `FightIQ_improved/`
- Current unified codebase: `fightiq_codex/`
If you have additional Kaggle project notes/schemas not in the repo, please share and I will integrate them into the data contracts and scrapers.
---
## 0) Principles & Constraints
- Leak‑safe by construction: strict temporal splits, point‑in‑time feature store, target/method/round leakage detectors.
- Reproducibility: pinned artifacts, signed runs, deterministic configs, versioned data.
- Calibration first: probabilities must be well‑calibrated before any betting policy tuning.
- Realistic ROI: vig removal, configurable slippage, event‑level compounding, exposure caps.
- Safety: scraping within ToS; secrets isolated; graceful retries and fallbacks.
## 1) Data Contracts & Feature Store
Goal: Single, versioned source of truth for all inputs and features with PTI (point‑in‑time) guarantees.
- Raw layer schemas (parquet):
  - `events_raw`: event_id, event_date, event_name, location, …
  - `fights_raw`: fight_id, event_id, fighter_1_id, fighter_2_id, scheduled_rounds, weight_class, title_fight, …
  - `fighters_raw`: fighter_id, name, stance, reach, height, dob, team, …
  - `round_stats_raw`: fight_id, round, f1/f2 per‑round stats (strikes att/succ, ctrl, TD, SUB, KD, …)
  - `odds_raw`: fight_id, book, timestamp, market (moneyline/method/round), side, price (decimal), is_open/close
  - `rankings_raw`: timestamped per‑division rankings
  - `weighins_raw`: event_id, fighter_id, scale, notes
  - `news_raw`: source_id, timestamp, text, URL (optional text features)
- Silver layer (cleaned/normalized):
  - Deduping, type fixes, canonical IDs, timezones, missing filling policies.
- Feature store (gold):
  - Pre‑fight tabular: rolling/expanding aggregates strictly from fights < current fight date; matchup deltas; bio deltas; recency features; rankings deltas; odds features (open/close/consensus) as configured; optional text/news embeddings.
  - Targets: winner (binary), method (multiclass: KO/TKO/SUB/DEC/OTHER), round (ordinal or finish time bins).
  - PTI join checks (unit tests + Great Expectations); target leakage scanners (winner/method/round keywords, regex exclusions).
- Validation:
  - Great Expectations suites per table; CI smoke on samples.
  - Drift checks: feature distributions vs rolling baselines.
Deliverables: data contracts (YAML/Markdown), ETL jobs, feature registry manifest, validation suites.
## 2) Scraping & Ingestion (Agent Kit)
Goal: Agent‑driven scrapers for UFC Stats/ESPN/Topology and odds; normalize to raw layer.
- Agents/tools:
  - HTTP/Playwright fetch; robots.txt aware; rotating headers; backoff/retry; change detection.
- Sources & endpoints:
  - Events + fight cards (dates, location, card order)
  - Fighter pages (bio, reach, stance, team)
  - Bout stats (per‑round + totals) – historical only (live model later)
  - Odds (open/close lines; moneyline/method/round)
  - Rankings history; weigh‑ins; news/injuries (optional)
- Schedules:
  - Nightly backfills for historical updates.
  - Weekly “upcoming card” scrape (Mon/Tue), refresh close odds (Fri/Sat).
- QA:
  - Contracts enforcement; counters; diffs vs previous snapshot; alerts on anomalies.
Deliverables: scraper modules, normalizers, job specs, ingestion tests.
## 3) Feature Engineering
Goal: Rich, leak‑safe, interpretable features.
- Rolling/expanding per‑fighter aggregates (3/5/10/15 fights): per‑round stats aggregated historically.
- Matchup deltas (fighter vs opponent): reach/height/age deltas; stance matchups; style proxies.
- Recency: time since last fight; camp changes; travel/timezone; altitude (if available).
- Rankings: division rank trend deltas; title fight flag.
- Odds: configurable inclusion (open/close/vig‑removed); market move features.
- Text: optional embeddings from weigh‑ins/news (injury/ring rust proxies).
- Segment tags: weight class, gender, title fight.
Deliverables: `feature_store/` transformers, PTI tests, profiling notebook.
## 4) Modeling – Multi‑Task + Ensembling
Goal: State‑of‑the‑art tabular baselines + calibrated, robust ensembling.
- Targets:
  - Winner (binary), Method (multiclass), Round (ordinal bins or discrete 1–5 + decisions)
- Baselines:
  - Tuned XGBoost/LightGBM/CatBoost per task (Optuna time‑series CV)
- Multi‑task model:
  - Shared encoder (tabular) with 3 heads; joint/weighted losses; per‑task calibration
- Ensembling:
  - OOF stacking: base models’ OOF preds → meta‑learner (LR/Calibrated LR)
  - Per‑segment models/stackers (by weight class) when data suffices
- Monotonicity where helpful (odds/implied prob vs win prob) for stability
Deliverables: `scripts/`scripts/t`scripts/train_multitask.py```, HPO pipelines, stacking utilities.
## 5) Calibration
Goal: Trustworthy probabilities powering ROI decisions.
- Winner: per‑segment (weight class) Platt/Isotonic; fallback global
- Method: temperature scaling or Dirichlet calibration
- Round: ordinal calibration (per boundary) or per‑class isotonic
- Metrics: ECE, Brier; accept gates per segment; monitoring
Deliverables: calibrators per task/segment; validation reports.
## 6) Evaluation & ROI Engine
Goal: Realistic backtests; event‑level compounding; robust policy tuning.
- Walk‑forward over years with time‑series CV; per‑year test windows
- Betting policy tuning (validation only):
  - Edge threshold, probability threshold
  - Kelly fraction cap; partial Kelly
  - Max bets per event; max exposure per event
  - Slippage scenario; open vs close odds sensitivity
- Live odds options (future): in‑fight updates for live model
- Reports: accuracy/logloss/AUC/ECE; ROI/hit/Sharpe/drawdown; equity curves; per‑segment breakdowns
Deliverables: `scripts/backtest_walkforward.py`, `scripts/evaluate_yearly_bets.py`, reports in `outputs/`.
## 7) Champion Selection & Registry
Goal: Automatic model promotion with safety gates.
- Gate metrics: min accuracy, max logloss, ECE caps, ROI targets
- Registry: champion/previous; artifacts with signatures; rollback
- Diff reports: what improved/regressed and by how much
Deliverables: registry module, promotion CLI, diff reports.
## 8) Weekly Automation & Deployment (Agent Kit)
Goal: Agent‑operated weekly loop.
- Orchestration flow (Mon–Sat):
  1) Scrape upcoming card; ingest odds
  2) Validate data quality; PTI checks
  3) Build features; drift check
  4) Train candidates (budgeted HPO) → calibrate → evaluate
  5) If passes gates → promote champion
  6) Predict upcoming fights → generate risk‑controlled bet plan
  7) Publish: CSV/JSON; Slack/Discord; dashboard update
- Artifacts: versioned predictions, bet plans, PDF/HTML summaries
- Monitoring & alerts: failures, drifts, performance flags
Deliverables: Agent Kit workflows, schedule definitions, integration to Slack/Email.
## 9) Dashboards & Reporting
Goal: Transparent, reproducible insights.
- Streamlit/Gradio app: recent accuracy, calibration plots, ROI curves
- Yearly summaries; per‑class (method/round) confusion and ECE
- Bet plan viewer with expected value and exposure per event
Deliverables: `app/` with Dockerfile, CI deploy to internal host.
## 10) Security, Compliance, Infra
- Respect ToS for scraping, avoid prohibited sources
- Secrets via .env/secret manager; rotated keys
- Dockerized jobs; pinned versions; CI smoke tests
- Storage: data lake (parquet), model registry, MLflow/W&B
## Roadmap & Milestones
### Phase A (Week 1–2): Stabilize & Extend
- [ ] Formalize data contracts (raw/silver/gold) with schemas & GE suites
- [ ] Build PTI joins for gold; implement leakage scanners for winner/method/round
- [ ] Wire tuned LGB/XGB/Cat HPO (Optuna) with time‑series CV (winner)
- [ ] Add per‑weight‑class calibration for winner (Platt/Isotonic) & monitoring
- [ ] Backtest winner with event‑level compounding; finalize risk controls
### Phase B (Week 3–4): Multi‑Task + Stacking
- [ ] Add method and round targets; extend feature store
- [ ] Train multi‑task model (shared encoder + 3 heads)
- [ ] Per‑task calibration (binary/multiclass/ordinal)
- [ ] OOF stacking/blending per segment; champion selection policy
### Phase C (Week 5–6): Scraping & Agent Workflows
- [ ] Scraper agents for UFC Stats/ESPN/Topology & odds; normalize to raw layer
- [ ] Weekly orchestrations with Agent Kit; add retries, backoffs, rate control
- [ ] Great Expectations checks & drift alerts; Slack notifications
### Phase D (Week 7–8): ROI & Policies
- [ ] Policy tuning (edge/prob thresholds; caps; partial Kelly) on val; simulate slippage
- [ ] Live odds support (close vs open) sensitivity; report impact
- [ ] Add per‑event exposure caps in planner; publish weekly bet PDFs/CSVs
### Phase E (Week 9–10): Dashboard & Deployment
- [ ] Streamlit dashboard; deploy to internal host
- [ ] Champion/registry tooling; rollback buttons and run diffs
- [ ] Final end‑to‑end test; sign‑off gates
## Specific TODOs (Execution Backlog)
### Data & Features
- [ ] Extract notebook logic from `FightIQ/` (Kaggle lineage) into ingestion/ETL modules
- [ ] Codify raw→silver→gold transforms; add PTI unit/integration tests
- [ ] Implement opponent/matchup deltas; recency decay features; rankings deltas
- [ ] Add text embeddings (optional) and toggle via config
### Modeling
- [ ] Winner: extended HPO grids; monotonic constraints on odds/implied features
- [ ] Method: multiclass baselines + temperature/Dirichlet calibration
- [ ] Round: ordinal/logit models + ordinal calibration
- [ ] Multi‑task model; loss weighting; early stopping via time‑CV
- [ ] OOF stacking (per segment) with calibrated meta‑learner
### Calibration & Eval
- [ ] Per‑segment calibrators; ECE/Brier gates; reporting
- [ ] Backtest extensions for method/round tasks and joint evaluation
- [ ] Policy tuning grid (edge/prob/kelly/constraints/slippage); event‑level compounding
### Ops & Agent Kit
- [ ] Scraper agent (UFC Stats/ESPN/Topology); odds fetcher; rankings/weigh‑ins; news
- [ ] Validation agent (GE + drift); auto‑issue creation
- [ ] Training agent (HPO, stacking, calibrators); promotion logic
- [ ] Deployment agent (predictions + bet planner + publishing)
- [ ] Orchestrator agent (weekly schedule, rollback)
### Reporting
- [ ] Streamlit app (accuracy, calibration, ROI, equity); per‑segment tables
- [ ] Slack digests; PDF export of weekly bet plan and rationale
## Notes on Round‑by‑Round Stats
- Pre‑fight model: Allowed only as historical aggregates (strictly prior fights). Current‑fight round stats are leakage.
- Live model (future): Separate pipeline that ingests live per‑round stats mid‑fight to update win/method/round predictions.
## Dependencies & Inputs to Confirm
- Kaggle notebooks & data lineage: confirm any additional schemas/cleaning steps beyond what’s in `FightIQ/`.
- Odds coverage & fields: confirm books/timepoints (open/close), and any consensus data you’d like integrated.
- Deployment targets: preferred scheduler (Airflow/Prefect) and hosting for dashboard.
## Acceptance Criteria
- PTI‑verified features; zero target leakage confirmed by tests.
- Calibrated probabilities with ECE ≤ thresholds per weight class.
- Backtests: stable accuracy/logloss + positive, risk‑controlled ROI across 4+ recent years.
- Automated weekly run with artifacted predictions, bet plan, and reports.
- Reproducible artifacts; one‑command replays; clear rollback.
## Current Status (Codex)
- Unified codebase exists with tuned LGB/XGB, per‑segment calibration, event‑level compounding, and ROI backtests.
- Yearly simulations (2022–2025) with £1000 start and per‑event compounding are generated under `fightiq_codex/outputs/yearly/` and summarized in `yearly_summary.csv`.
## Next Immediate Steps
- Implement raw→silver→gold contracts referencing existing `FightIQ/` notebooks.
- Add method/round targets and extend training/backtesting to multi‑task with calibration.
- Start scraper agent stubs and integrate with feature store.
## Activity Log
2025‑10‑20
- Added ingestion scaffolding: Parquet/BigQuery sinks and events/fight URLs/fight stats scrapers; scripts: `ingest_events.py`, `ingest_fight_urls.py`, `ingest_fight_stats.py`.
- Built fights_silver and odds_silver builders; ingest odds/rankings from CSV; odds API client + ingest script.
- Added rankings_silver (rank as of event_date, per fighter) and updated fight stats scraper to extract weight_class.
- Built first gold features (PTI rolling means over prior 5 fights + matchup deltas + vig‑free odds + rankings deltas) → `data/gold_features.parquet`.
- Yearly event‑level compounding simulation with £1000 start across 2022–2025; bet CSVs and equity per year; summary written.
2025‑10‑20 (cont.)
- Added lightweight data validations script: `scripts/validate_data.py` for raw/silver/gold sanity checks.
- Implemented initial multi‑task training scaffold: `scripts/`scripts/t`scripts/train_multitask.py``` (winner/method/round baselines using gold features + labels from original golden dataset).
- Added multi-task backtest summary via time-series CV: `scripts/backtest_multitask.py` → outputs `outputs/backtest_multitask_summary.csv`.
- Added weekly orchestration stub: `scripts/`scripts/weekly_orchestrate.py`` (ingest → build → validate → train), ready to wrap in Agent Kit.
- Implemented simple calibrators: Winner Platt (per head) and temperature scaling for multiclass; hooked into ``scripts/t`scripts/train_multitask.py``` to save calibrators.
- Integrated per-fold calibration into backtest (winner Platt; method/round temperature scaling); wrote calibrated CV metrics.
- Added GE runner stub `scripts/run_ge_validations.py` to execute expectations when GE is installed.
Planned next:
- Great Expectations validations for raw/silver; CI smoke on a subset. (GE runner stub added; install `great-expectations` to enable.)
- Multi‑task targets (winner/method/round) in gold and training scripts.
- Agent Kit orchestration for weekly scrape→validate→build→train→calibrate→publish.
2025-10-21
- Added parity winner script `scripts/train_winner_parity.py` to retrain tuned LightGBM/XGBoost on the original golden dataset using leak-safe splits and imputation.
- Ran parity training; LightGBM reached 73.6% val accuracy and XGBoost 74.0% (logloss ~0.52). Metrics saved to `outputs/parity_winner_summary.csv`; artifacts published under `artifacts/parity_winner/20251021_094843`.
- Enhanced gold features with recency/experience metrics (days since fight, fights in horizons, avg downtime), extended rolling windows (20 fights), odds vig/logit features, rank aggregates; rebuilt `data/gold_features.parquet` (warnings noted about pandas fragmentation).
- Added `scripts/train_winner_enhanced.py` to train tuned LightGBM/XGBoost + logistic stack on enriched gold features; validation accuracy ~0.73 (logloss 0.53) and test accuracy ~0.69 (logloss 0.62). Metrics stored in `outputs/winner_enhanced_summary.csv`; artifacts saved under `artifacts/winner_enhanced/20251021_103820`.
- Defined Great Expectations suites in src/validation/ge_suites.py and wired scripts/run_ge_validations.py to run them (skips gracefully if GE missing); updated weekly orchestrator to invoke GE and new winner training.
- Installed Great Expectations (0.17.21), refreshed GE runner to use suites and generated outputs/ge_summary.json; current failures highlight 1 missing raw odds fight_url and out-of-range ranks (0) in raw/silver feeds.
- Refactored gold feature builder to batch concat rolling features (eliminates pandas fragmentation warnings).
- Added guard rails: `scripts/train_winner_enhanced.py` now accepts accuracy thresholds (default stack ensemble) and exits if val/test accuracy drop below configured floors; meta JSON records guard metrics. `scripts/weekly_orchestrate.py` invokes GE checks, retrains with guard thresholds (0.72/0.68), and reruns parity training.
- Updated `scripts/train_multitask.py` to merge in leak-safe original features, report CV aggregates to `outputs/multitask_train_summary.json`, and reuse the new feature store (winner ~0.69 acc, method ~0.84, round ~0.999).
- Cleaned ingestion: odds CSV drops blank/'nan' fight URLs (rewrite rather than merge) and rankings <=0 set to NA; src/ingestion/sinks.ParquetSink now supports replace writes and sanitizes null keys. GE suite passes with outputs/ge_summary.json.
- Tightened leakage defenses: expanded loader leak patterns (r1_duration, takedown stats) and re-trained multi-task models (round acc ~0.57) plus winner guard thresholds reset (0.71/0.62).
- Added scripts/ci_smoke.py to run GE, guarded winner training, and parity thresholds (0.72/0.60) for CI automation; updated weekly orchestrator guard thresholds.
- Extended scripts/predict_upcoming.py to add multitask method/round probabilities (loading rtifacts/multitask features/calibrators) alongside existing winner + Kelly columns.
- Tuned LightGBM/XGBoost on enriched gold set (scripts/tune_lgb_enhanced.py, scripts/tune_xgb_enhanced.py); updated config to new params and retrained winner ensemble (stack val acc 0.726, test acc 0.633).
- Integrated parity LightGBM predictions into winner stack (kept base models untouched, inject parity as third meta-feature); 	rain_winner_enhanced.py loads parity bundle, blends in stacking, and predict_upcoming.py surfaces parity probabilities.
- Added config-driven auto loading for latest zero-importance feature lists; `train_winner_enhanced.py` now consumes them without manual flagging and retraining restored stack metrics (val acc 0.734, test acc 0.638).
- Re-ran winner enhanced training (20251021_153832) with parity feature injection + feature pruning; artifacts/outputs refreshed and guard metrics recorded in meta.json.
- Enhanced `train_multitask.py` with class-weighted LightGBM, early stopping, raw-logit temperature scaling, and shared drop-feature automation; regenerated multitask artifacts + richer summary JSON.
- Hardened `predict_upcoming.py` to resolve winner artifact runs under artifacts/winner_enhanced, support meta.json, and apply calibrated method/round probabilities with normalization.
- Added regression tests (`tests/test_multitask_outputs.py`) verifying calibrated method/round probability vectors remain normalized given latest artifacts.
- Built `scripts/ingest_upcoming_from_odds_api.py` to pull consensus odds via TheOddsAPI key, normalize fighter/order metadata, and materialize `FightIQ/data/upcoming_fights.csv` + raw snapshots. Added richer odds metadata (vig-free probabilities, book coverage).
- Updated loader/imputer utilities: `align_to_training_features` now uses DataFrame reindex (no fragmentation) and imputer skips duplicating `_missing` indicators when columns already aligned for inference.
- Verified end-to-end prediction flow on freshly pulled 25-Oct-2025 card (parity + stack + multitask outputs) saving results under `fightiq_codex/outputs/upcoming_predictions_*.csv`.
- Generated full PTI feature matrix for upcoming fights via `scripts/build_upcoming_features.py`; outputs `fightiq_codex/data/upcoming_features.parquet` aligned with training schema (f1/f2, deltas, odds, rankings). `predict_upcoming.py` now consumes this richer dataset.
- Weekly orchestrator now calls upcoming odds ingestion + feature build when `THEODDS_API_KEY` is present, ensuring Agent Kit flow has ready-to-predict data each run.
