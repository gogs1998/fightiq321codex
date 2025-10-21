"""
Walk-forward backtest with fold-level calibration and EV-aware betting.

Features:
- Uses tuned LightGBM params if provided in config.
- Splits each training fold into inner-train and inner-calibration sets to fit a calibrator.
- Tunes min_edge, Kelly cap, and Kelly multiplier on the calibration set; freezes for test.
- Removes vig from odds if enabled; supports arbitrary odds field names from config.
- Places bets on the side with positive EV (f1 or f2), applying constraints.
- Writes per-fold metrics and an aggregate ROI report.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.loaders import UFCDataLoader
from src.data.splitters import WalkForwardSplitter
from src.data.preprocessing import FeatureTypeImputationStrategy
from src.evaluation.metrics import MetricsCalculator
from src.betting.kelly_criterion import kelly_fraction
from src.models.calibration import PlattCalibrator, IsotonicCalibrator
from src.models.ensemble import StackingEnsemble
import json

# Helpers
def _simulate_roi(p_proba, y_true, o1, o2, imp1, imp2, valid_mask, min_edge, cap, k_mult, prob_threshold, constraints: dict):
    bet_frac, profit = _place_bets(p_proba, y_true, o1, o2, imp1, imp2, valid_mask, min_edge, cap, k_mult, prob_threshold, constraints, None)
    tw = float(np.sum(bet_frac))
    roi = float(np.sum(profit) / tw) if tw > 0 else -1e9
    # Sharpe per bet
    per_bet = profit / np.clip(bet_frac, 1e-9, None)
    sharpe = float(np.mean(per_bet) / (np.std(per_bet) + 1e-9)) if tw > 0 else -1e9
    return roi, sharpe


def _place_bets(p1, y_true, o1, o2, imp1, imp2, valid_mask, min_edge, cap, k_mult, prob_threshold, constraints: dict, test_rows: pd.DataFrame | None):
    # EV per side
    edge1 = p1 * o1 - 1.0
    p2 = 1.0 - p1
    edge2 = p2 * o2 - 1.0

    f1 = kelly_fraction(p1, o1) * k_mult
    f2 = kelly_fraction(p2, o2) * k_mult

    # Only bet if edge >= min_edge; select side with higher edge
    bet_frac = np.zeros_like(p1)
    side = np.where((edge1 >= min_edge) & (edge1 >= edge2), 1, 0)
    side = np.where((edge2 >= min_edge) & (edge2 > edge1), 2, side)

    # Apply caps and validity
    f_sel = np.where(side == 1, np.minimum(f1, cap), np.where(side == 2, np.minimum(f2, cap), 0.0))
    f_sel = np.where(valid_mask, f_sel, 0.0)
    # Probability threshold on selected side
    p_side = np.where(side == 1, p1, np.where(side == 2, 1.0 - p1, 0.0))
    f_sel = np.where(p_side >= prob_threshold, f_sel, 0.0)

    # Diversity constraints per event (limit bets per event; and cap exposure per event)
    if test_rows is not None:
        max_bets_per_event = constraints.get("max_bets_per_event", 999)
        max_exposure_per_event = constraints.get("max_exposure_per_event", None)
        if "event_name" in test_rows.columns:
            bet_idx = np.where(f_sel > 0)[0]
            events = test_rows["event_name"].astype(str).values
            used = {}
            for idx in bet_idx:
                ev = events[idx]
                cnt = used.get(ev, 0)
                if cnt >= max_bets_per_event:
                    f_sel[idx] = 0.0
                else:
                    used[ev] = cnt + 1
            # Exposure cap: scale down bets within an event if sum exceeds cap
            if isinstance(max_exposure_per_event, (int, float)) and max_exposure_per_event > 0:
                unique_events = {}
                for idx in np.where(f_sel > 0)[0]:
                    ev = events[idx]
                    unique_events.setdefault(ev, []).append(idx)
                for ev, idxs in unique_events.items():
                    s = float(np.sum(f_sel[idxs]))
                    if s > max_exposure_per_event:
                        scale = max_exposure_per_event / s
                        f_sel[idxs] = f_sel[idxs] * scale

    # Profit per fight based on chosen side
    result = (y_true == 1).astype(int)
    profit = np.where(side == 1, np.where(result == 1, f_sel * (o1 - 1.0), -f_sel), 0.0)
    profit += np.where(side == 2, np.where(result == 0, f_sel * (o2 - 1.0), -f_sel), 0.0)
    return f_sel, profit


def _aggregate_roi(profits: np.ndarray, bets: np.ndarray) -> dict:
    total_wager = float(np.sum(bets))
    total_profit = float(np.sum(profits))
    roi = float(total_profit / total_wager) if total_wager > 0 else 0.0
    # Hit rate on placed bets
    wins = (profits > 0).astype(int)
    hit_rate = float(np.mean(wins[bets > 0])) if np.any(bets > 0) else 0.0
    # Returns per bet and Sharpe (per bet; not annualized)
    per_bet_ret = profits / np.clip(bets, 1e-9, None)
    sharpe = float(np.mean(per_bet_ret) / (np.std(per_bet_ret) + 1e-9))
    # Max drawdown across cumulative profits
    cum = np.cumsum(profits)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    max_dd = float(np.min(drawdown))
    return {
        "total_wager": total_wager,
        "total_profit": total_profit,
        "roi": roi,
        "hit_rate": hit_rate,
        "sharpe_per_bet": sharpe,
        "max_drawdown": max_dd,
        "n_bets": int(np.sum(bets > 0)),
    }


def _fit_segment_calibrators(y_proba: np.ndarray, y_true: np.ndarray, segments: pd.Series | None, method: str):
    # Global calibrator
    if method == "platt":
        global_cal = PlattCalibrator().fit(y_proba, y_true)
    else:
        global_cal = IsotonicCalibrator().fit(y_proba, y_true)

    cal_map = {}
    if segments is not None:
        seg_vals = segments.astype(str)
        for seg in seg_vals.unique():
            mask = (seg_vals == seg).values
            if mask.sum() >= 30:  # require min samples
                yp = y_proba[mask]
                yt = y_true[mask]
                try:
                    if method == "platt":
                        cal_map[seg] = PlattCalibrator().fit(yp, yt)
                    else:
                        cal_map[seg] = IsotonicCalibrator().fit(yp, yt)
                except Exception:
                    # fallback silently
                    pass
    return cal_map, global_cal


def _apply_segment_calibration(y_proba: np.ndarray, segments: pd.Series | None, cal_map: dict, global_cal):
    if segments is None or len(cal_map) == 0:
        return global_cal.transform(y_proba)
    seg_vals = segments.astype(str)
    y_cal = np.empty_like(y_proba)
    for i in range(len(y_proba)):
        seg = seg_vals.iloc[i]
        cal = cal_map.get(seg, global_cal)
        y_cal[i] = cal.transform(np.array([y_proba[i]]))[0]
    return y_cal


def main():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    loader = UFCDataLoader(cfg["paths"]["data_dir"])
    df = loader.load_golden_dataset(cfg["paths"]["golden_dataset"])

    wf = WalkForwardSplitter(
        initial_train_end="2020-12-31",
        final_test_end="2024-12-31",
        test_window_months=3,
        step_months=1,
        min_train_size=1000,
    )
    folds = wf.create_folds(df)
    # Evaluate only most recent 12 folds for speed
    if len(folds) > 12:
        folds = folds[-12:]

    metrics = MetricsCalculator()
    records = []
    all_profits = []
    all_bets = []

    # Load tuned LGB params if available
    tuned_params = None
    tuned_path = cfg.get("modeling", {}).get("tuned_lgb_params_path")
    if tuned_path and Path(tuned_path).exists():
        tuned_params = json.loads(Path(tuned_path).read_text())

    calibrator_method = cfg.get("modeling", {}).get("calibrator", "platt").lower()

    for i, (train_df, test_df) in enumerate(folds, start=1):
        Xtr_raw, ytr = loader.prepare_features_target(train_df, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"])
        Xte_raw, yte = loader.prepare_features_target(test_df, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"])

        # For speed and fewer columns in backtest, disable indicators here
        imputer = FeatureTypeImputationStrategy(create_indicators=False).fit(Xtr_raw)
        Xtr_full = imputer.transform(Xtr_raw)
        Xte = imputer.transform(Xte_raw)

        # Inner split: last 10% of training fold as calibration set
        split_idx = int(len(Xtr_full) * 0.9)
        Xtr = Xtr_full.iloc[:split_idx]
        ytr_inner = ytr.iloc[:split_idx]
        Xcal = Xtr_full.iloc[split_idx:]
        ycal = ytr.iloc[split_idx:]

        # Train candidate models and pick best by calibrated calibration loss (with segment calibration)
        candidates = []
        # LightGBM candidate
        lgb_params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.08, "num_leaves": 31, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1}
        if tuned_params:
            lgb_params.update(tuned_params)
        mdl_lgb = lgb.train(lgb_params, lgb.Dataset(Xtr, label=ytr_inner), num_boost_round=300, callbacks=[lgb.log_evaluation(period=0)])
        proba_cal_lgb = mdl_lgb.predict(Xcal)
        seg_cal_map_lgb, global_cal_lgb = _fit_segment_calibrators(proba_cal_lgb, ycal.values, train_df.loc[ycal.index, "weight_class"] if "weight_class" in train_df.columns else None, calibrator_method)
        proba_calibrated_lgb = _apply_segment_calibration(proba_cal_lgb, train_df.loc[ycal.index, "weight_class"] if "weight_class" in train_df.columns else None, seg_cal_map_lgb, global_cal_lgb)
        # use Brier as a proxy evaluation on calibration slice
        cal_loss_lgb = float(np.mean((proba_calibrated_lgb - ycal.values) ** 2))
        candidates.append(("lgb", mdl_lgb, (seg_cal_map_lgb, global_cal_lgb), cal_loss_lgb))

        # XGBoost candidate (if tuned params exist)
        tuned_xgb_path = cfg.get("modeling", {}).get("tuned_xgb_params_path")
        if tuned_xgb_path and Path(tuned_xgb_path).exists():
            xgb_params = json.loads(Path(tuned_xgb_path).read_text())
            dtr = xgb.DMatrix(Xtr, label=ytr_inner)
            mdl_xgb = xgb.train(xgb_params, dtr, num_boost_round=300)
            pcal_xgb = mdl_xgb.predict(xgb.DMatrix(Xcal))
            seg_cal_map_xgb, global_cal_xgb = _fit_segment_calibrators(pcal_xgb, ycal.values, train_df.loc[ycal.index, "weight_class"] if "weight_class" in train_df.columns else None, calibrator_method)
            pcal_xgb_cal = _apply_segment_calibration(pcal_xgb, train_df.loc[ycal.index, "weight_class"] if "weight_class" in train_df.columns else None, seg_cal_map_xgb, global_cal_xgb)
            cal_loss_xgb = float(np.mean((pcal_xgb_cal - ycal.values) ** 2))
            candidates.append(("xgb", mdl_xgb, (seg_cal_map_xgb, global_cal_xgb), cal_loss_xgb))

        # Stacking ensemble candidate if tuned params exist
        if tuned_params and tuned_xgb_path and Path(tuned_xgb_path).exists():
            xgb_params = json.loads(Path(tuned_xgb_path).read_text())
            def _trainer_xgb(Xa, ya):
                d = xgb.DMatrix(Xa, label=ya)
                return xgb.train(xgb_params, d, num_boost_round=300)
            def _trainer_lgb(Xa, ya):
                d = lgb.Dataset(Xa, label=ya)
                return lgb.train(lgb_params, d, num_boost_round=300, callbacks=[lgb.log_evaluation(period=0)])
            base = [
                {"name": "xgb", "trainer": _trainer_xgb},
                {"name": "lgb", "trainer": _trainer_lgb},
            ]
            ens = StackingEnsemble(base_models=base, n_splits=5)
            ens.fit(Xtr, ytr_inner)
            pcal_ens = ens.predict_proba(Xcal)
            seg_cal_map_ens, global_cal_ens = _fit_segment_calibrators(pcal_ens, ycal.values, train_df.loc[ycal.index, "weight_class"] if "weight_class" in train_df.columns else None, calibrator_method)
            pcal_ens_cal = _apply_segment_calibration(pcal_ens, train_df.loc[ycal.index, "weight_class"] if "weight_class" in train_df.columns else None, seg_cal_map_ens, global_cal_ens)
            cal_loss_ens = float(np.mean((pcal_ens_cal - ycal.values) ** 2))
            candidates.append(("ens", ens, (seg_cal_map_ens, global_cal_ens), cal_loss_ens))

        name, model, calibrator, _ = min(candidates, key=lambda t: t[3])

        # Predict on test fold and calibrate
        if name == "lgb":
            yproba_raw = model.predict(Xte)
        elif name == "xgb":
            yproba_raw = model.predict(xgb.DMatrix(Xte))
        else:
            # ensemble
            yproba_raw = model.predict_proba(Xte)
        seg_map, global_cal = calibrator
        yproba = _apply_segment_calibration(yproba_raw, test_df.loc[yte.index, "weight_class"] if "weight_class" in test_df.columns else None, seg_map, global_cal)
        ypred = (yproba >= 0.5).astype(int)

        cls = metrics.classification(yte.values, ypred, yproba)
        cal = metrics.calibration(yte.values, yproba)

        rec = {"fold": i, **cls, **cal, "n_test": int(len(yte))}

        # Optional ROI if odds available
        f1_field = cfg["betting"]["odds_fields"].get("f1", "f_1_odds")
        f2_field = cfg["betting"]["odds_fields"].get("f2", "f_2_odds")
        if {f1_field, f2_field}.issubset(test_df.columns):
            o1_series = test_df.loc[yte.index, f1_field].astype(float)
            o2_series = test_df.loc[yte.index, f2_field].astype(float)
            valid_mask = o1_series.notna().values
            o1 = o1_series.fillna(0).values
            o2 = o2_series.fillna(0).values

            # Remove vig if enabled
            if cfg["betting"].get("remove_vig", True):
                imp1 = 1.0 / np.clip(o1, 1e-9, None)
                imp2 = 1.0 / np.clip(o2, 1e-9, None)
                s = imp1 + imp2
                imp1 /= np.where(s > 0, s, 1.0)
                imp2 /= np.where(s > 0, s, 1.0)
            else:
                imp1 = 1.0 / np.clip(o1, 1e-9, None)
                imp2 = 1.0 / np.clip(o2, 1e-9, None)

            # Tune bet policy on inner calibration set using available odds (fallback to defaults if missing)
            min_edge_grid = cfg["betting"]["tuning"]["min_edge_grid"]
            cap_grid = cfg["betting"]["tuning"]["kelly_cap_grid"]
            mult_grid = cfg["betting"]["tuning"]["kelly_multiplier_grid"]
            prob_grid = cfg["betting"]["tuning"].get("prob_threshold_grid", [0.0])
            best_roi, best_policy = -1e9, (cfg["betting"]["min_edge"], cfg["betting"]["kelly_fraction_cap"], cfg["betting"].get("kelly_multiplier", 1.0), 0.0)
            best_sharpe = -1e9
            objective = cfg["betting"]["tuning"].get("objective", "roi").lower()

            # Need calibration odds for tuning; use the same event-level odds fields if present
            if {f1_field, f2_field}.issubset(train_df.columns):
                oc1 = train_df.loc[ycal.index, f1_field].astype(float)
                oc2 = train_df.loc[ycal.index, f2_field].astype(float)
                vmask_c = oc1.notna().values
                oc1 = oc1.fillna(0).values
                oc2 = oc2.fillna(0).values
                if name == "lgb":
                    pcal_raw = model.predict(Xcal)
                elif name == "xgb":
                    pcal_raw = model.predict(xgb.DMatrix(Xcal))
                else:
                    pcal_raw = model.predict_proba(Xcal)
                seg_map_sel, global_cal_sel = calibrator
                pcal = _apply_segment_calibration(pcal_raw, train_df.loc[ycal.index, "weight_class"] if "weight_class" in train_df.columns else None, seg_map_sel, global_cal_sel)
                # Remove vig for calibration odds if enabled
                if cfg["betting"].get("remove_vig", True):
                    ic1 = 1.0 / np.clip(oc1, 1e-9, None)
                    ic2 = 1.0 / np.clip(oc2, 1e-9, None)
                    ss = ic1 + ic2
                    ic1 /= np.where(ss > 0, ss, 1.0)
                    ic2 /= np.where(ss > 0, ss, 1.0)
                else:
                    ic1 = 1.0 / np.clip(oc1, 1e-9, None)
                    ic2 = 1.0 / np.clip(oc2, 1e-9, None)

                for me in min_edge_grid:
                    for cap in cap_grid:
                        for km in mult_grid:
                            for pth in prob_grid:
                                roi_c, sharpe_c = _simulate_roi(pcal, ycal.values, oc1, oc2, ic1, ic2, vmask_c, me, cap, km, pth, constraints=cfg["betting"].get("constraints", {}))
                                score = roi_c if objective == "roi" else sharpe_c
                                best_score = best_roi if objective == "roi" else best_sharpe
                                if score > best_score:
                                    best_roi = roi_c
                                    best_sharpe = sharpe_c
                                    best_policy = (me, cap, km, pth)

            min_edge, cap, k_mult, prob_threshold = best_policy

            # EV-based side selection and bet sizing on test
            bet_frac, profit = _place_bets(yproba, yte.values, o1, o2, imp1, imp2, valid_mask, min_edge, cap, k_mult, prob_threshold, cfg["betting"].get("constraints", {}), test_df.loc[yte.index])
            total_wager = float(np.sum(bet_frac))
            roi = float(np.sum(profit) / total_wager) if total_wager > 0 else 0.0
            rec.update({"roi": roi, "total_bets": int(np.sum(bet_frac > 0))})
            all_profits.extend(profit.tolist())
            all_bets.extend(bet_frac.tolist())

        records.append(rec)

    df_rec = pd.DataFrame(records)
    out = ROOT / "outputs" / "walkforward_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_rec.to_csv(out, index=False)
    logger.info(f"Saved walk-forward summary to {out}")

    # Aggregate ROI report
    if all_bets:
        agg = _aggregate_roi(np.array(all_profits), np.array(all_bets))
        rep = ROOT / "outputs" / "walkforward_roi_report.json"
        rep.write_text(json.dumps(agg, indent=2))
        logger.info(f"Saved ROI aggregate report to {rep}")


if __name__ == "__main__":
    main()
