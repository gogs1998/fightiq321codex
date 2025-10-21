"""
Evaluate 2025 holdout accuracy and ROI using tuned models with calibration and EV/kelly tuning.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from loguru import logger
import lightgbm as lgb
import xgboost as xgb

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.loaders import UFCDataLoader
from src.data.splitters import TemporalSplitter
from src.data.preprocessing import FeatureTypeImputationStrategy
from src.evaluation.metrics import MetricsCalculator
from src.models.calibration import PlattCalibrator, IsotonicCalibrator
from src.models.ensemble import StackingEnsemble


def _fit_segment_calibrators(y_proba: np.ndarray, y_true: np.ndarray, segments: pd.Series | None, method: str):
    if method == "platt":
        global_cal = PlattCalibrator().fit(y_proba, y_true)
    else:
        global_cal = IsotonicCalibrator().fit(y_proba, y_true)
    cal_map = {}
    if segments is not None:
        seg_vals = segments.astype(str)
        for seg in seg_vals.unique():
            mask = (seg_vals == seg).values
            if mask.sum() >= 30:
                yp = y_proba[mask]
                yt = y_true[mask]
                try:
                    if method == "platt":
                        cal_map[seg] = PlattCalibrator().fit(yp, yt)
                    else:
                        cal_map[seg] = IsotonicCalibrator().fit(yp, yt)
                except Exception:
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


def _place_bets(p1, y_true, o1, o2, valid_mask, min_edge, cap, k_mult, prob_threshold, constraints: dict, test_rows: pd.DataFrame | None):
    edge1 = p1 * o1 - 1.0
    p2 = 1.0 - p1
    edge2 = p2 * o2 - 1.0
    # Kelly fractions
    b1 = o1 - 1.0
    f1 = (p1 * (b1 + 1) - 1) / np.where(b1 != 0, b1, 1e-9)
    f1 = np.clip(np.maximum(f1, 0.0), 0.0, 1.0) * k_mult
    b2 = o2 - 1.0
    f2 = (p2 * (b2 + 1) - 1) / np.where(b2 != 0, b2, 1e-9)
    f2 = np.clip(np.maximum(f2, 0.0), 0.0, 1.0) * k_mult

    bet_frac = np.zeros_like(p1)
    side = np.where((edge1 >= min_edge) & (edge1 >= edge2), 1, 0)
    side = np.where((edge2 >= min_edge) & (edge2 > edge1), 2, side)
    f_sel = np.where(side == 1, np.minimum(f1, cap), np.where(side == 2, np.minimum(f2, cap), 0.0))
    f_sel = np.where(valid_mask, f_sel, 0.0)
    p_side = np.where(side == 1, p1, np.where(side == 2, 1.0 - p1, 0.0))
    f_sel = np.where(p_side >= prob_threshold, f_sel, 0.0)

    if test_rows is not None and "event_name" in test_rows.columns:
        max_bets_per_event = constraints.get("max_bets_per_event", 999)
        max_exposure_per_event = constraints.get("max_exposure_per_event", None)
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
        if isinstance(max_exposure_per_event, (int, float)) and max_exposure_per_event > 0:
            groups = {}
            for idx in np.where(f_sel > 0)[0]:
                ev = events[idx]
                groups.setdefault(ev, []).append(idx)
            for ev, idxs in groups.items():
                s = float(np.sum(f_sel[idxs]))
                if s > max_exposure_per_event:
                    scale = max_exposure_per_event / s
                    f_sel[idxs] = f_sel[idxs] * scale

    result = (y_true == 1).astype(int)
    profit = np.where(side == 1, np.where(result == 1, f_sel * (o1 - 1.0), -f_sel), 0.0)
    profit += np.where(side == 2, np.where(result == 0, f_sel * (o2 - 1.0), -f_sel), 0.0)
    return f_sel, profit


def main():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    loader = UFCDataLoader(cfg["paths"]["data_dir"])
    df = loader.load_golden_dataset(cfg["paths"]["golden_dataset"])
    splitter = TemporalSplitter(cfg["splits"]["val_start_date"], cfg["splits"]["test_start_date"])
    split = splitter.split(df)

    # Prepare features
    Xtr_raw, ytr = loader.prepare_features_target(split.train, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"])
    Xva_raw, yva = loader.prepare_features_target(split.val, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"])
    Xte_raw, yte = loader.prepare_features_target(split.test, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"])

    imputer = FeatureTypeImputationStrategy(create_indicators=False).fit(Xtr_raw)
    Xtr = imputer.transform(Xtr_raw)
    Xva = imputer.transform(Xva_raw)
    Xte = imputer.transform(Xte_raw)

    # Train candidates on train; evaluate/calibrate on val; choose best; evaluate on test
    tuned_lgb_path = cfg.get("modeling", {}).get("tuned_lgb_params_path")
    tuned_xgb_path = cfg.get("modeling", {}).get("tuned_xgb_params_path")
    lgb_params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.08, "num_leaves": 31, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1}
    if tuned_lgb_path and Path(tuned_lgb_path).exists():
        lgb_params.update(json.loads(Path(tuned_lgb_path).read_text()))

    candidates = []
    # LGB
    mdl_lgb = lgb.train(lgb_params, lgb.Dataset(Xtr, label=ytr), num_boost_round=400, callbacks=[lgb.log_evaluation(period=0)])
    pva_lgb = mdl_lgb.predict(Xva)
    seg_map_lgb, global_cal_lgb = _fit_segment_calibrators(pva_lgb, yva.values, split.val["weight_class"] if "weight_class" in split.val.columns else None, cfg.get("modeling", {}).get("calibrator", "platt"))
    pva_lgb_cal = _apply_segment_calibration(pva_lgb, split.val["weight_class"] if "weight_class" in split.val.columns else None, seg_map_lgb, global_cal_lgb)
    cal_loss_lgb = float(np.mean((pva_lgb_cal - yva.values) ** 2))
    candidates.append(("lgb", mdl_lgb, (seg_map_lgb, global_cal_lgb), cal_loss_lgb))

    # XGB
    if tuned_xgb_path and Path(tuned_xgb_path).exists():
        xgb_params = json.loads(Path(tuned_xgb_path).read_text())
        mdl_xgb = xgb.train(xgb_params, xgb.DMatrix(Xtr, label=ytr), num_boost_round=400)
        pva_xgb = mdl_xgb.predict(xgb.DMatrix(Xva))
        seg_map_xgb, global_cal_xgb = _fit_segment_calibrators(pva_xgb, yva.values, split.val["weight_class"] if "weight_class" in split.val.columns else None, cfg.get("modeling", {}).get("calibrator", "platt"))
        pva_xgb_cal = _apply_segment_calibration(pva_xgb, split.val["weight_class"] if "weight_class" in split.val.columns else None, seg_map_xgb, global_cal_xgb)
        cal_loss_xgb = float(np.mean((pva_xgb_cal - yva.values) ** 2))
        candidates.append(("xgb", mdl_xgb, (seg_map_xgb, global_cal_xgb), cal_loss_xgb))

    # Ensemble
    if tuned_xgb_path and Path(tuned_xgb_path).exists():
        def _trainer_xgb(Xa, ya):
            return xgb.train(json.loads(Path(tuned_xgb_path).read_text()), xgb.DMatrix(Xa, label=ya), num_boost_round=400)
        def _trainer_lgb(Xa, ya):
            return lgb.train(lgb_params, lgb.Dataset(Xa, label=ya), num_boost_round=400, callbacks=[lgb.log_evaluation(period=0)])
        ens = StackingEnsemble(base_models=[{"name": "xgb", "trainer": _trainer_xgb}, {"name": "lgb", "trainer": _trainer_lgb}], n_splits=5)
        ens.fit(Xtr, ytr)
        pva_ens = ens.predict_proba(Xva)
        seg_map_ens, global_cal_ens = _fit_segment_calibrators(pva_ens, yva.values, split.val["weight_class"] if "weight_class" in split.val.columns else None, cfg.get("modeling", {}).get("calibrator", "platt"))
        pva_ens_cal = _apply_segment_calibration(pva_ens, split.val["weight_class"] if "weight_class" in split.val.columns else None, seg_map_ens, global_cal_ens)
        cal_loss_ens = float(np.mean((pva_ens_cal - yva.values) ** 2))
        candidates.append(("ens", ens, (seg_map_ens, global_cal_ens), cal_loss_ens))

    name, model, calibrators, _ = min(candidates, key=lambda t: t[3])
    seg_map, global_cal = calibrators

    # Predict on 2025 test
    if name == "lgb":
        pte_raw = model.predict(Xte)
    elif name == "xgb":
        pte_raw = model.predict(xgb.DMatrix(Xte))
    else:
        pte_raw = model.predict_proba(Xte)
    pte = _apply_segment_calibration(pte_raw, split.test["weight_class"] if "weight_class" in split.test.columns else None, seg_map, global_cal)

    # Accuracy & metrics
    metrics = MetricsCalculator()
    ypred = (pte >= 0.5).astype(int)
    cls = metrics.classification(yte.values, ypred, pte)

    # ROI on 2025
    f1_field = cfg["betting"]["odds_fields"].get("f1", "f_1_odds")
    f2_field = cfg["betting"]["odds_fields"].get("f2", "f_2_odds")
    roi = {"roi": 0.0, "n_bets": 0}
    if {f1_field, f2_field}.issubset(split.test.columns):
        o1_series = split.test.loc[yte.index, f1_field].astype(float)
        o2_series = split.test.loc[yte.index, f2_field].astype(float)
        valid = o1_series.notna().values
        o1 = o1_series.fillna(0).values
        o2 = o2_series.fillna(0).values
        # Remove vig if enabled
        if cfg["betting"].get("remove_vig", True):
            imp1 = 1.0 / np.clip(o1, 1e-9, None)
            imp2 = 1.0 / np.clip(o2, 1e-9, None)
            s = imp1 + imp2
            imp1 /= np.where(s > 0, s, 1.0)
            imp2 /= np.where(s > 0, s, 1.0)
        # Tune policy on validation and apply to test
        min_edge_grid = cfg["betting"]["tuning"]["min_edge_grid"]
        cap_grid = cfg["betting"]["tuning"]["kelly_cap_grid"]
        mult_grid = cfg["betting"]["tuning"]["kelly_multiplier_grid"]
        prob_grid = cfg["betting"]["tuning"].get("prob_threshold_grid", [0.0])
        best = (-1e9, None)
        objective = cfg["betting"]["tuning"].get("objective", "roi").lower()
        # Validation tuning
        if {f1_field, f2_field}.issubset(split.val.columns):
            oc1 = split.val.loc[yva.index, f1_field].astype(float).fillna(0).values
            oc2 = split.val.loc[yva.index, f2_field].astype(float).fillna(0).values
            if cfg["betting"].get("remove_vig", True):
                ic1 = 1.0 / np.clip(oc1, 1e-9, None)
                ic2 = 1.0 / np.clip(oc2, 1e-9, None)
                ss = ic1 + ic2
                ic1 /= np.where(ss > 0, ss, 1.0)
                ic2 /= np.where(ss > 0, ss, 1.0)
            else:
                ic1 = 1.0 / np.clip(oc1, 1e-9, None)
                ic2 = 1.0 / np.clip(oc2, 1e-9, None)
            pva_sel = pva_lgb_cal if name == "lgb" else (pva_xgb_cal if name == "xgb" else pva_ens_cal)
            vmask_c = split.val.loc[yva.index, f1_field].notna().values
            for me in min_edge_grid:
                for cap in cap_grid:
                    for km in mult_grid:
                        for pth in prob_grid:
                            f_sel, prof = _place_bets(pva_sel, yva.values, oc1, oc2, vmask_c, me, cap, km, pth, cfg["betting"].get("constraints", {}), split.val.loc[yva.index])
                            tw = float(np.sum(f_sel))
                            if tw > 0:
                                roi_v = float(np.sum(prof) / tw)
                                per_bet = prof / np.clip(f_sel, 1e-9, None)
                                sharpe = float(np.mean(per_bet) / (np.std(per_bet) + 1e-9))
                                score = roi_v if objective == "roi" else sharpe
                                if score > best[0]:
                                    best = (score, (me, cap, km, pth))
        # Apply best policy on test
        me, cap, km, pth = best[1] if best[1] is not None else (cfg["betting"]["min_edge"], cfg["betting"]["kelly_fraction_cap"], cfg["betting"].get("kelly_multiplier", 1.0), 0.0)
        f_sel, prof = _place_bets(pte, yte.values, o1, o2, valid, me, cap, km, pth, cfg["betting"].get("constraints", {}), split.test.loc[yte.index])
        tw = float(np.sum(f_sel))
        roi = {
            "roi": float(np.sum(prof) / tw) if tw > 0 else 0.0,
            "n_bets": int(np.sum(f_sel > 0)),
            "total_wager": tw,
            "total_profit": float(np.sum(prof)),
        }

        # Bankroll simulation starting at £100
        initial_bankroll = 100.0
        bankroll = initial_bankroll
        equity = [bankroll]
        # Determine chosen side for each bet
        p1 = pte
        p2 = 1.0 - p1
        edge1 = p1 * o1 - 1.0
        edge2 = p2 * o2 - 1.0
        side = np.where((edge1 >= me) & (edge1 >= edge2), 1, 0)
        side = np.where((edge2 >= me) & (edge2 > edge1), 2, side)
        # Kelly fractions per side (capped + prob threshold applied like in f_sel)
        b1 = o1 - 1.0
        k1 = np.clip(np.maximum((p1 * (b1 + 1) - 1) / np.where(b1 != 0, b1, 1e-9), 0.0), 0.0, 1.0) * km
        b2 = o2 - 1.0
        k2 = np.clip(np.maximum((p2 * (b2 + 1) - 1) / np.where(b2 != 0, b2, 1e-9), 0.0), 0.0, 1.0) * km
        frac = np.where(side == 1, np.minimum(k1, cap), np.where(side == 2, np.minimum(k2, cap), 0.0))
        # Apply validity and prob threshold
        p_side = np.where(side == 1, p1, np.where(side == 2, p2, 0.0))
        frac = np.where(valid & (p_side >= pth), frac, 0.0)

        # Simulate sequentially (sorted by date already)
        for i in range(len(frac)):
            f = float(frac[i])
            if f <= 0: 
                equity.append(bankroll)
                continue
            wager = bankroll * f
            if side[i] == 1:
                if yte.values[i] == 1:
                    bankroll += wager * (o1[i] - 1.0)
                else:
                    bankroll -= wager
            elif side[i] == 2:
                if yte.values[i] == 0:
                    bankroll += wager * (o2[i] - 1.0)
                else:
                    bankroll -= wager
            equity.append(bankroll)

        equity = np.array(equity)
        returns = equity[1:] / np.clip(equity[:-1], 1e-9, None) - 1.0
        max_run = np.maximum.accumulate(equity)
        drawdown = equity - max_run
        # Annualize based on days span in test
        d0 = pd.to_datetime(split.test['event_date'].min())
        d1 = pd.to_datetime(split.test['event_date'].max())
        days = max((d1 - d0).days, 1)
        total_ret = bankroll / initial_bankroll - 1.0
        cagr = (bankroll / initial_bankroll) ** (365.0 / days) - 1.0 if bankroll > 0 else -1.0
        bankroll_report = {
            'initial': initial_bankroll,
            'final': float(bankroll),
            'total_return': float(total_ret),
            'cagr': float(cagr),
            'max_drawdown': float(drawdown.min()),
            'sharpe_per_step': float(returns.mean() / (returns.std() + 1e-9)) if len(returns) > 0 else 0.0,
            'steps': int(len(returns))
        }

    # Report
    print("Holdout 2025 accuracy/logloss/auc:", cls)
    print("Holdout 2025 ROI:", roi)
    if 'bankroll_report' in locals():
        print("Holdout 2025 bankroll (initial £100):", bankroll_report)


if __name__ == "__main__":
    main()
