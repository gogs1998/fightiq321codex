"""
Train tuned winner models on the original golden dataset to confirm baseline parity.

This script mirrors the original setup:
  - Loads the historic FightIQ golden CSV (leak-safe features).
  - Applies the standard temporal split (train < val < test).
  - Fits the feature-type imputer with missing indicators.
  - Trains the tuned LightGBM and XGBoost models using the saved best params.
  - Reports validation/test metrics (logloss/accuracy/ROC AUC + calibration stats).
  - Saves artifacts under artifacts/parity_winner/<timestamp>.
  - Writes a summary CSV to outputs/parity_winner_summary.csv.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger

ROOT = Path(__file__).parents[1]

import sys

sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.loaders import UFCDataLoader
from src.data.splitters import TemporalSplitter
from src.data.preprocessing import FeatureTypeImputationStrategy
from src.data.validation import DataValidator
from src.evaluation.metrics import MetricsCalculator
from src.models.calibration import PlattCalibrator


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    for base in [ROOT, ROOT.parent]:
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def _load_params(path: str | Path | None) -> Dict:
    if not path:
        return {}
    p = _resolve_path(path)
    if not p.exists():
        logger.warning(f"Tuned params not found at {p}; falling back to defaults.")
        return {}
    with p.open("r", encoding="utf-8") as fh:
        params = json.load(fh)
    return params


def _ensure_float32(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(np.float32)


def train_lightgbm(params: Dict, Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray):
    base_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "verbose": -1,
    }
    base_params.update(params or {})
    train_set = lgb.Dataset(Xtr, label=ytr)
    val_set = lgb.Dataset(Xva, label=yva, reference=train_set)
    callbacks = [lgb.log_evaluation(period=0), lgb.early_stopping(50, verbose=False)]
    booster = lgb.train(
        base_params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    best_iter = booster.best_iteration or booster.current_iteration()
    return booster, best_iter


def train_xgboost(params: Dict, Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray):
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "tree_method": "hist",
    }
    base_params.update(params or {})
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dval = xgb.DMatrix(Xva, label=yva)
    booster = xgb.train(
        base_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    best_iter = booster.best_iteration or booster.num_boosted_rounds()
    return booster, best_iter


def compute_split_metrics(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metrics: MetricsCalculator,
    calibrated: bool = False,
) -> Dict[str, float | str]:
    if len(y_true) == 0:
        return {}
    y_pred = (y_proba >= 0.5).astype(int)
    cls = metrics.classification(y_true, y_pred, y_proba)
    cal = metrics.calibration(y_true, y_proba)
    row: Dict[str, float | str] = {
        "model": model_name,
        "split": split_name,
        "calibrated": "yes" if calibrated else "no",
    }
    row.update({k: float(v) for k, v in cls.items()})
    row.update(cal)
    return row


def main():
    cfg = load_config(ROOT / "config" / "config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    data_dir = _resolve_path(cfg["paths"]["data_dir"])
    loader = UFCDataLoader(str(data_dir))
    golden_path = _resolve_path(cfg["paths"]["golden_dataset"])
    df = loader.load_golden_dataset(str(golden_path))

    validator = DataValidator(df)
    validation_result = validator.validate_all()
    if not validation_result["passed"]:
        for err in validation_result["errors"]:
            logger.error(err)
        raise RuntimeError("Data validation failed; parity training aborted.")
    for warn in validation_result["warnings"]:
        logger.warning(warn)

    splitter = TemporalSplitter(cfg["splits"]["val_start_date"], cfg["splits"]["test_start_date"])
    split = splitter.split(df)

    Xtr_raw, ytr = loader.prepare_features_target(
        split.train, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"]
    )
    Xva_raw, yva = loader.prepare_features_target(
        split.val, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"]
    )
    Xte_raw, yte = loader.prepare_features_target(
        split.test, target="winner_encoded", exclude_odds=cfg["features"]["exclude_odds"]
    )

    # Fit imputer on training data only
    imputer = FeatureTypeImputationStrategy(create_indicators=cfg["features"]["create_missing_indicators"]).fit(Xtr_raw)
    Xtr = _ensure_float32(imputer.transform(Xtr_raw))
    Xva = _ensure_float32(imputer.transform(Xva_raw))
    Xte = _ensure_float32(imputer.transform(Xte_raw))

    ytr_arr = ytr.values.astype(int)
    yva_arr = yva.values.astype(int)
    yte_arr = yte.values.astype(int) if len(yte) > 0 else np.array([], dtype=int)

    metrics = MetricsCalculator()
    rows: List[Dict[str, float | str]] = []
    artifacts_root = ROOT / "artifacts" / "parity_winner"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = artifacts_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist shared artifacts
    joblib.dump({"features": Xtr.columns.tolist()}, run_dir / "features.pkl")
    joblib.dump(imputer, run_dir / "imputer.pkl")

    tuned_lgb_params = _load_params(cfg.get("modeling", {}).get("tuned_lgb_params_path"))
    tuned_xgb_params = _load_params(cfg.get("modeling", {}).get("tuned_xgb_params_path"))

    if tuned_lgb_params:
        logger.info("Training tuned LightGBM model for parity check...")
        lgb_model, lgb_best_iter = train_lightgbm(tuned_lgb_params, Xtr, ytr_arr, Xva, yva_arr)
        joblib.dump({"params": tuned_lgb_params, "best_iteration": lgb_best_iter}, run_dir / "lightgbm_meta.pkl")
        lgb_model.save_model(str(run_dir / "lightgbm_model.txt"))

        proba_val = lgb_model.predict(Xva, num_iteration=lgb_best_iter)
        rows.append(compute_split_metrics("lightgbm_tuned", "val", yva_arr, proba_val, metrics))

        if len(yte_arr) > 0:
            proba_test = lgb_model.predict(Xte, num_iteration=lgb_best_iter)
            rows.append(compute_split_metrics("lightgbm_tuned", "test", yte_arr, proba_test, metrics))

            calibrator = PlattCalibrator().fit(proba_val, yva_arr)
            joblib.dump(calibrator, run_dir / "lightgbm_calibrator.pkl")
            proba_test_cal = calibrator.transform(proba_test)
            rows.append(
                compute_split_metrics("lightgbm_tuned", "test", yte_arr, proba_test_cal, metrics, calibrated=True)
            )
        else:
            logger.warning("Test split empty; skipping LightGBM test metrics.")
    else:
        logger.warning("Skipping LightGBM parity training; tuned params unavailable.")

    if tuned_xgb_params:
        logger.info("Training tuned XGBoost model for parity check...")
        xgb_model, xgb_best_iter = train_xgboost(tuned_xgb_params, Xtr, ytr_arr, Xva, yva_arr)
        xgb_model.save_model(str(run_dir / "xgboost_model.json"))
        joblib.dump({"params": tuned_xgb_params, "best_iteration": xgb_best_iter}, run_dir / "xgboost_meta.pkl")

        dval = xgb.DMatrix(Xva)
        proba_val = xgb_model.predict(dval, iteration_range=(0, xgb_best_iter))
        rows.append(compute_split_metrics("xgboost_tuned", "val", yva_arr, proba_val, metrics))

        if len(yte_arr) > 0:
            dtest = xgb.DMatrix(Xte)
            proba_test = xgb_model.predict(dtest, iteration_range=(0, xgb_best_iter))
            rows.append(compute_split_metrics("xgboost_tuned", "test", yte_arr, proba_test, metrics))

            calibrator = PlattCalibrator().fit(proba_val, yva_arr)
            joblib.dump(calibrator, run_dir / "xgboost_calibrator.pkl")
            proba_test_cal = calibrator.transform(proba_test)
            rows.append(
                compute_split_metrics("xgboost_tuned", "test", yte_arr, proba_test_cal, metrics, calibrated=True)
            )
        else:
            logger.warning("Test split empty; skipping XGBoost test metrics.")
    else:
        logger.warning("Skipping XGBoost parity training; tuned params unavailable.")

    if rows:
        summary_path = ROOT / "outputs" / "parity_winner_summary.csv"
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Wrote parity metrics to {summary_path}")
    else:
        logger.error("No models were trained; parity summary not created.")


if __name__ == "__main__":
    main()
