"""
Export a detailed list of 2025 holdout bets using the calibrated, tuned policy.
Overrides: min_edge=0.04, kelly_cap=0.02, kelly_multiplier=0.25.
Writes CSV to fightiq_codex/outputs/holdout_2025_bets.csv
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


def main():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    loader = UFCDataLoader(cfg["paths"]["data_dir"])
    df = loader.load_golden_dataset(cfg["paths"]["golden_dataset"])
    splitter = TemporalSplitter(cfg["splits"]["val_start_date"], cfg["splits"]["test_start_date"])
    split = splitter.split(df)

    Xtr_raw, ytr = loader.prepare_features_target(split.train, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"])
    Xva_raw, yva = loader.prepare_features_target(split.val, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"])
    Xte_raw, yte = loader.prepare_features_target(split.test, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"])

    imputer = FeatureTypeImputationStrategy(create_indicators=False).fit(Xtr_raw)
    Xtr = imputer.transform(Xtr_raw)
    Xva = imputer.transform(Xva_raw)
    Xte = imputer.transform(Xte_raw)

    tuned_lgb_path = cfg.get("modeling", {}).get("tuned_lgb_params_path")
    tuned_xgb_path = cfg.get("modeling", {}).get("tuned_xgb_params_path")
    lgb_params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.08, "num_leaves": 31, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1}
    if tuned_lgb_path and Path(tuned_lgb_path).exists():
        lgb_params.update(json.loads(Path(tuned_lgb_path).read_text()))

    candidates = []
    # LGB
    mdl_lgb = lgb.train(lgb_params, lgb.Dataset(Xtr, label=ytr), num_boost_round=400, callbacks=[lgb.log_evaluation(period=0)])
    pva_lgb = mdl_lgb.predict(Xva)
    seg_map_lgb, global_cal_lgb = fit_segment_calibrators(pva_lgb, yva.values, split.val["weight_class"] if "weight_class" in split.val.columns else None, cfg.get("modeling", {}).get("calibrator", "platt"))
    pva_lgb_cal = apply_segment_calibration(pva_lgb, split.val["weight_class"] if "weight_class" in split.val.columns else None, seg_map_lgb, global_cal_lgb)
    cal_loss_lgb = float(np.mean((pva_lgb_cal - yva.values) ** 2))
    candidates.append(("lgb", mdl_lgb, (seg_map_lgb, global_cal_lgb), cal_loss_lgb))

    # XGB
    if tuned_xgb_path and Path(tuned_xgb_path).exists():
        xgb_params = json.loads(Path(tuned_xgb_path).read_text())
        mdl_xgb = xgb.train(xgb_params, xgb.DMatrix(Xtr, label=ytr), num_boost_round=400)
        pva_xgb = mdl_xgb.predict(xgb.DMatrix(Xva))
        seg_map_xgb, global_cal_xgb = fit_segment_calibrators(pva_xgb, yva.values, split.val["weight_class"] if "weight_class" in split.val.columns else None, cfg.get("modeling", {}).get("calibrator", "platt"))
        pva_xgb_cal = apply_segment_calibration(pva_xgb, split.val["weight_class"] if "weight_class" in split.val.columns else None, seg_map_xgb, global_cal_xgb)
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
        seg_map_ens, global_cal_ens = fit_segment_calibrators(pva_ens, yva.values, split.val["weight_class"] if "weight_class" in split.val.columns else None, cfg.get("modeling", {}).get("calibrator", "platt"))
        pva_ens_cal = apply_segment_calibration(pva_ens, split.val["weight_class"] if "weight_class" in split.val.columns else None, seg_map_ens, global_cal_ens)
        cal_loss_ens = float(np.mean((pva_ens_cal - yva.values) ** 2))
        candidates.append(("ens", ens, (seg_map_ens, global_cal_ens), cal_loss_ens))

    name, model, calibrators, _ = min(candidates, key=lambda t: t[3])
    seg_map, global_cal = calibrators

    # Predict calibrated probabilities on 2025 test
    if name == "lgb":
        pte_raw = model.predict(Xte)
    elif name == "xgb":
        pte_raw = model.predict(xgb.DMatrix(Xte))
    else:
        pte_raw = model.predict_proba(Xte)
    pte = apply_segment_calibration(pte_raw, split.test["weight_class"] if "weight_class" in split.test.columns else None, seg_map, global_cal)

    # Build bets list with overrides
    f1_field = cfg["betting"]["odds_fields"].get("f1", "f_1_odds")
    f2_field = cfg["betting"]["odds_fields"].get("f2", "f_2_odds")
    assert {f1_field, f2_field}.issubset(split.test.columns), "Odds fields not present in test set."
    o1 = split.test.loc[Xte.index, f1_field].astype(float).fillna(0).values
    o2 = split.test.loc[Xte.index, f2_field].astype(float).fillna(0).values
    valid = split.test.loc[Xte.index, f1_field].notna().values

    # Overrides
    me, cap, km, pth = 0.04, 0.02, 0.25, cfg["betting"]["tuning"].get("prob_threshold_grid", [0.0])[0]

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
    p_side = np.where(side == 1, p1, np.where(side == 2, p2, 0.0))
    f_sel = np.where(valid & (p_side >= pth), f_sel, 0.0)

    # Build rows
    test_idx = Xte.index
    rows = []
    result = (yte.values == 1).astype(int)
    for i, idx in enumerate(test_idx):
        if f_sel[i] <= 0:
            continue
        event_date = split.test.loc[idx, 'event_date'] if 'event_date' in split.test.columns else None
        event_name = split.test.loc[idx, 'event_name'] if 'event_name' in split.test.columns else None
        wclass = split.test.loc[idx, 'weight_class'] if 'weight_class' in split.test.columns else None
        f1n = split.test.loc[idx, 'f_1_name'] if 'f_1_name' in split.test.columns else None
        f2n = split.test.loc[idx, 'f_2_name'] if 'f_2_name' in split.test.columns else None
        # Per-unit stake return (profit if staking 1.0 unit on chosen side)
        if side[i] == 1:
            unit_profit = (o1[i] - 1.0) if result[i] == 1 else -1.0
        else:
            unit_profit = (o2[i] - 1.0) if result[i] == 0 else -1.0
        rows.append({
            'event_date': str(event_date) if event_date is not None else None,
            'event_name': event_name,
            'weight_class': wclass,
            'fighter_1': f1n,
            'fighter_2': f2n,
            'chosen_side': int(side[i]),
            'p_f1': float(p1[i]),
            'p_f2': float(p2[i]),
            'odds_f1': float(o1[i]),
            'odds_f2': float(o2[i]),
            'edge_f1': float(edge1[i]),
            'edge_f2': float(edge2[i]),
            'bet_fraction': float(f_sel[i]),
            'outcome_f1_won': int(result[i]),
            'unit_profit': float(unit_profit),
        })

    out_csv = ROOT / 'outputs' / 'holdout_2025_bets.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Wrote bets list:", out_csv)


if __name__ == "__main__":
    main()
