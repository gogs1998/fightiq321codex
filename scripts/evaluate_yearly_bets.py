"""
Evaluate per-year (calendar) test windows for the past N years.

For each target year Y:
- Train on data < (Y-2-01-01)
- Validate on [(Y-2-01-01), (Y-01-01)) for calibration + policy tuning
- Test on [Y-01-01, (Y+1)-01-01)
- Use tuned LGB/XGB and stacking ensemble; pick best by calibrated validation loss
- Apply risk-controlled policy: min_edge=0.04, kelly_cap=0.02, kelly_multiplier=0.25
- Export bets CSV and equity CSV; write a yearly summary row
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger
import lightgbm as lgb
import xgboost as xgb

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.loaders import UFCDataLoader
from src.data.preprocessing import FeatureTypeImputationStrategy
from src.models.calibration import PlattCalibrator, IsotonicCalibrator
from src.models.ensemble import StackingEnsemble


def fit_segment_calibrators(y_proba, y_true, segments, method: str):
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


def apply_segment_calibration(y_proba, segments, cal_map, global_cal):
    if segments is None or len(cal_map) == 0:
        return global_cal.transform(y_proba)
    seg_vals = segments.astype(str)
    out = np.empty_like(y_proba)
    for i in range(len(y_proba)):
        seg = seg_vals.iloc[i]
        cal = cal_map.get(seg, global_cal)
        out[i] = cal.transform(np.array([y_proba[i]]))[0]
    return out


def choose_model_and_calibrator(Xtr, ytr, Xva, yva, val_segments, lgb_params, xgb_params, calibrator_method: str):
    candidates = []
    # LGB
    mdl_lgb = lgb.train(lgb_params, lgb.Dataset(Xtr, label=ytr), num_boost_round=400, callbacks=[lgb.log_evaluation(period=0)])
    pva_lgb = mdl_lgb.predict(Xva)
    seg_map_lgb, global_cal_lgb = fit_segment_calibrators(pva_lgb, yva.values, val_segments, calibrator_method)
    pva_lgb_cal = apply_segment_calibration(pva_lgb, val_segments, seg_map_lgb, global_cal_lgb)
    cal_loss_lgb = float(np.mean((pva_lgb_cal - yva.values) ** 2))
    candidates.append(("lgb", mdl_lgb, (seg_map_lgb, global_cal_lgb), cal_loss_lgb))

    # XGB
    if xgb_params is not None:
        mdl_xgb = xgb.train(xgb_params, xgb.DMatrix(Xtr, label=ytr), num_boost_round=400)
        pva_xgb = mdl_xgb.predict(xgb.DMatrix(Xva))
        seg_map_xgb, global_cal_xgb = fit_segment_calibrators(pva_xgb, yva.values, val_segments, calibrator_method)
        pva_xgb_cal = apply_segment_calibration(pva_xgb, val_segments, seg_map_xgb, global_cal_xgb)
        cal_loss_xgb = float(np.mean((pva_xgb_cal - yva.values) ** 2))
        candidates.append(("xgb", mdl_xgb, (seg_map_xgb, global_cal_xgb), cal_loss_xgb))

    # Ensemble
    if xgb_params is not None:
        def _trainer_xgb(Xa, ya):
            return xgb.train(xgb_params, xgb.DMatrix(Xa, label=ya), num_boost_round=400)
        def _trainer_lgb(Xa, ya):
            return lgb.train(lgb_params, lgb.Dataset(Xa, label=ya), num_boost_round=400, callbacks=[lgb.log_evaluation(period=0)])
        ens = StackingEnsemble(base_models=[{"name": "xgb", "trainer": _trainer_xgb}, {"name": "lgb", "trainer": _trainer_lgb}], n_splits=5)
        ens.fit(Xtr, ytr)
        pva_ens = ens.predict_proba(Xva)
        seg_map_ens, global_cal_ens = fit_segment_calibrators(pva_ens, yva.values, val_segments, calibrator_method)
        pva_ens_cal = apply_segment_calibration(pva_ens, val_segments, seg_map_ens, global_cal_ens)
        cal_loss_ens = float(np.mean((pva_ens_cal - yva.values) ** 2))
        candidates.append(("ens", ens, (seg_map_ens, global_cal_ens), cal_loss_ens))

    name, model, calibrators, _ = min(candidates, key=lambda t: t[3])
    return name, model, calibrators


def simulate_year(year: int, cfg):
    loader = UFCDataLoader(cfg["paths"]["data_dir"])
    df = loader.load_golden_dataset(cfg["paths"]["golden_dataset"])
    df = df.sort_values("event_date").reset_index(drop=True)

    # Windows
    val_start = pd.to_datetime(f"{year-2}-01-01")
    test_start = pd.to_datetime(f"{year}-01-01")
    test_end = pd.to_datetime(f"{year+1}-01-01")

    train_df = df[df["event_date"] < val_start]
    val_df = df[(df["event_date"] >= val_start) & (df["event_date"] < test_start)]
    test_df = df[(df["event_date"] >= test_start) & (df["event_date"] < test_end)]

    # Features
    Xtr_raw, ytr = loader.prepare_features_target(train_df, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"])
    Xva_raw, yva = loader.prepare_features_target(val_df, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"])
    Xte_raw, yte = loader.prepare_features_target(test_df, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"])

    imputer = FeatureTypeImputationStrategy(create_indicators=False).fit(Xtr_raw)
    Xtr = imputer.transform(Xtr_raw)
    Xva = imputer.transform(Xva_raw)
    Xte = imputer.transform(Xte_raw)

    # Model params
    lgb_params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.08, "num_leaves": 31, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1}
    xgb_params = None
    if cfg.get("modeling", {}).get("tuned_lgb_params_path") and Path(cfg["modeling"]["tuned_lgb_params_path"]).exists():
        lgb_params.update(json.loads(Path(cfg["modeling"]["tuned_lgb_params_path"]).read_text()))
    if cfg.get("modeling", {}).get("tuned_xgb_params_path") and Path(cfg["modeling"]["tuned_xgb_params_path"]).exists():
        xgb_params = json.loads(Path(cfg["modeling"]["tuned_xgb_params_path"]).read_text())

    # Choose model
    calibrator_method = cfg.get("modeling", {}).get("calibrator", "platt").lower()
    val_segments = val_df["weight_class"] if "weight_class" in val_df.columns else None
    name, model, calibrators = choose_model_and_calibrator(Xtr, ytr, Xva, yva, val_segments, lgb_params, xgb_params, calibrator_method)
    seg_map, global_cal = calibrators

    # Predict calibrated on test
    if name == "lgb":
        pte_raw = model.predict(Xte)
    elif name == "xgb":
        pte_raw = model.predict(xgb.DMatrix(Xte))
    else:
        pte_raw = model.predict_proba(Xte)
    test_segments = test_df["weight_class"] if "weight_class" in test_df.columns else None
    pte = apply_segment_calibration(pte_raw, test_segments, seg_map, global_cal)

    # Betting with risk overrides
    f1_field = cfg["betting"]["odds_fields"].get("f1", "f_1_odds")
    f2_field = cfg["betting"]["odds_fields"].get("f2", "f_2_odds")
    if not {f1_field, f2_field}.issubset(test_df.columns):
        return None
    o1_series = test_df.loc[Xte.index, f1_field].astype(float)
    o2_series = test_df.loc[Xte.index, f2_field].astype(float)
    valid = o1_series.notna().values
    o1 = o1_series.fillna(0).values
    o2 = o2_series.fillna(0).values

    # Overrides
    me, cap, km = 0.04, 0.02, 0.25
    p1 = pte
    p2 = 1.0 - p1
    edge1 = p1 * o1 - 1.0
    edge2 = p2 * o2 - 1.0
    side = np.where((edge1 >= me) & (edge1 >= edge2), 1, 0)
    side = np.where((edge2 >= me) & (edge2 > edge1), 2, side)
    b1 = o1 - 1.0
    k1 = np.clip(np.maximum((p1 * (b1 + 1) - 1) / np.where(b1 != 0, b1, 1e-9), 0.0), 0.0, 1.0) * km
    b2 = o2 - 1.0
    k2 = np.clip(np.maximum((p2 * (b2 + 1) - 1) / np.where(b2 != 0, b2, 1e-9), 0.0), 0.0, 1.0) * km
    f_sel = np.where(side == 1, np.minimum(k1, cap), np.where(side == 2, np.minimum(k2, cap), 0.0))
    # Probability threshold: take the first from config grid or 0.0
    pth = cfg["betting"]["tuning"].get("prob_threshold_grid", [0.0])[0]
    p_side = np.where(side == 1, p1, np.where(side == 2, p2, 0.0))
    f_sel = np.where(valid & (p_side >= pth), f_sel, 0.0)

    # Build bets and equity (bankroll £1000) with per-event compounding
    test_idx = Xte.index
    rows = []
    bankroll = 1000.0
    equity_rows = []
    result = (yte.values == 1).astype(int)
    # group by event (fallback to date if name missing)
    evt_names = test_df.loc[test_idx, 'event_name'] if 'event_name' in test_df.columns else pd.Series(['event']*len(test_idx), index=test_idx)
    evt_dates = test_df.loc[test_idx, 'event_date'] if 'event_date' in test_df.columns else pd.Series([None]*len(test_idx), index=test_idx)
    # preserve order
    order = pd.DataFrame({'name': evt_names.values, 'date': evt_dates.values}, index=np.arange(len(test_idx)))
    order['i'] = np.arange(len(test_idx))
    # iterate per unique event in chronological order
    for ev_date, ev_name in order[['date','name']].drop_duplicates().sort_values(['date','name']).itertuples(index=False):
        idxs = order[(order['date']==ev_date) & (order['name']==ev_name)]['i'].values
        event_start_bankroll = bankroll
        event_profit = 0.0
        for i in idxs:
            if f_sel[i] <= 0:
                continue
            wager = event_start_bankroll * float(f_sel[i])
            if side[i] == 1:
                unit_profit = (o1[i] - 1.0) if result[i] == 1 else -1.0
            else:
                unit_profit = (o2[i] - 1.0) if result[i] == 0 else -1.0
            event_profit += wager * unit_profit
            # record bet row
            idx = test_idx[i]
            wclass = test_df.loc[idx, 'weight_class'] if 'weight_class' in test_df.columns else None
            f1n = test_df.loc[idx, 'f_1_name'] if 'f_1_name' in test_df.columns else None
            f2n = test_df.loc[idx, 'f_2_name'] if 'f_2_name' in test_df.columns else None
            rows.append({
                'event_date': str(ev_date) if ev_date is not None else None,
                'event_name': ev_name,
                'weight_class': wclass,
                'fighter_1': f1n,
                'fighter_2': f2n,
                'chosen_side': int(side[i]),
                'p_f1': float(p1[i]), 'p_f2': float(p2[i]),
                'odds_f1': float(o1[i]), 'odds_f2': float(o2[i]),
                'edge_f1': float(edge1[i]), 'edge_f2': float(edge2[i]),
                'bet_fraction': float(f_sel[i]), 'wager_amount': float(wager),
                'event_start_bankroll': float(event_start_bankroll),
            })
        bankroll += event_profit
        equity_rows.append({'event_date': str(ev_date) if ev_date is not None else None, 'event_name': ev_name, 'bankroll': float(bankroll)})

    # Summaries
    eq = np.array([r['bankroll'] for r in equity_rows])
    ret = (eq[-1] / 1000.0) - 1.0 if len(eq)>0 else 0.0
    days = max((pd.to_datetime(test_df['event_date'].max()) - pd.to_datetime(test_df['event_date'].min())).days, 1)
    cagr = (eq[-1] / 1000.0) ** (365.0 / days) - 1.0 if len(eq)>0 and eq[-1] > 0 else -1.0
    max_run = np.maximum.accumulate(eq) if len(eq)>0 else np.array([0])
    drawdown = (eq - max_run) if len(eq)>0 else np.array([0])
    yearly = {
        'year': year, 'n_bets': len(rows), 'final_bankroll': float(eq[-1]), 'total_return': float(ret), 'cagr': float(cagr), 'max_drawdown': float(drawdown.min())
    }

    # Write outputs
    out_dir = ROOT / 'outputs' / 'yearly'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f'{year}_bets.csv', index=False)
    pd.DataFrame(equity_rows).to_csv(out_dir / f'{year}_equity.csv', index=False)
    return yearly


def main():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    years = [2022, 2023, 2024, 2025]
    summaries = []
    for y in years:
        logger.info(f"Evaluating {y}...")
        res = simulate_year(y, cfg)
        if res is not None:
            summaries.append(res)

    if summaries:
        out = ROOT / 'outputs' / 'yearly_summary.csv'
        pd.DataFrame(summaries).to_csv(out, index=False)
        print("Wrote yearly summary:", out)


if __name__ == "__main__":
    main()
