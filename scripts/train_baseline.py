"""
Train baseline and optional ensemble models with calibration.
Config-driven and leak-safe. Saves artifacts for inference.
"""

import sys
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.loaders import UFCDataLoader
from src.data.splitters import TemporalSplitter
from src.data.preprocessing import FeatureTypeImputationStrategy
from src.data.validation import DataValidator
from src.evaluation.metrics import MetricsCalculator
from src.models.calibration import PlattCalibrator, IsotonicCalibrator
from src.models.ensemble import StackingEnsemble


def _train_logistic(Xtr, ytr):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(Xtr_s, ytr)
    return model, scaler


def _train_xgb(Xtr, ytr, params=None):
    params = params or {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "tree_method": "hist",
    }
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    model = xgb.train(params, dtrain, num_boost_round=500)
    return model


def _train_lgb(Xtr, ytr, params=None):
    params = params or {
        "objective": "binary",
        "metric": "binary_logloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "random_state": 42,
        "verbose": -1,
    }
    train_data = lgb.Dataset(Xtr, label=ytr)
    model = lgb.train(params, train_data, num_boost_round=500)
    return model


def main():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    # Load data
    loader = UFCDataLoader(cfg["paths"]["data_dir"])
    df = loader.load_golden_dataset(cfg["paths"]["golden_dataset"])

    # Validate
    val = DataValidator(df)
    res = val.validate_all()
    if not res["passed"]:
        for e in res["errors"]:
            logger.error(f"Validation error: {e}")
        raise RuntimeError("Data validation failed")

    # Split
    splitter = TemporalSplitter(
        cfg["splits"]["val_start_date"], cfg["splits"]["test_start_date"]
    )
    split = splitter.split(df)

    # Features/target
    Xtr_raw, ytr = loader.prepare_features_target(
        split.train,
        target="winner_encoded",
        exclude_odds=cfg["features"]["exclude_odds"],
    )
    Xva_raw, yva = loader.prepare_features_target(
        split.val,
        target="winner_encoded",
        exclude_odds=cfg["features"]["exclude_odds"],
    )

    # Optional sampling for quick runs
    sample_size = cfg["training"].get("sample_size")
    if sample_size:
        Xtr_raw = Xtr_raw.sample(sample_size, random_state=cfg["training"]["random_state"])  # type: ignore
        ytr = ytr.loc[Xtr_raw.index]

    # Imputation
    imputer = FeatureTypeImputationStrategy(
        create_indicators=cfg["features"]["create_missing_indicators"]
    ).fit(Xtr_raw)
    Xtr = imputer.transform(Xtr_raw)
    Xva = imputer.transform(Xva_raw)

    # Train candidates
    candidates = {}
    metrics = MetricsCalculator()

    # Logistic Regression
    if "logistic_regression" in cfg["model"]["candidates"]:
        lr_model, scaler = _train_logistic(Xtr, ytr)
        yva_proba = lr_model.predict_proba(scaler.transform(Xva))[:, 1]
        yva_pred = (yva_proba >= 0.5).astype(int)
        cand = {
            "name": "logistic_regression",
            "model": lr_model,
            "scaler": scaler,
            "val": metrics.classification(yva.values, yva_pred, yva_proba),
            "predict": lambda X: lr_model.predict_proba(scaler.transform(X))[:, 1],
        }
        candidates[cand["name"]] = cand

    # XGBoost
    if "xgboost" in cfg["model"]["candidates"]:
        xgb_model = _train_xgb(Xtr, ytr)
        yva_proba = xgb_model.predict(xgb.DMatrix(Xva))
        yva_pred = (yva_proba >= 0.5).astype(int)
        cand = {
            "name": "xgboost",
            "model": xgb_model,
            "scaler": None,
            "val": metrics.classification(yva.values, yva_pred, yva_proba),
            "predict": lambda X: xgb_model.predict(xgb.DMatrix(X)),
        }
        candidates[cand["name"]] = cand

    # LightGBM
    if "lightgbm" in cfg["model"]["candidates"]:
        lgb_model = _train_lgb(Xtr, ytr)
        yva_proba = lgb_model.predict(Xva, num_iteration=getattr(lgb_model, "best_iteration", None))
        yva_pred = (yva_proba >= 0.5).astype(int)
        cand = {
            "name": "lightgbm",
            "model": lgb_model,
            "scaler": None,
            "val": metrics.classification(yva.values, yva_pred, yva_proba),
            "predict": lambda X: lgb_model.predict(X, num_iteration=getattr(lgb_model, "best_iteration", None)),
        }
        candidates[cand["name"]] = cand

    # Choose best by validation log_loss
    best_name = min(candidates.keys(), key=lambda k: candidates[k]["val"]["log_loss"])  # type: ignore
    best = candidates[best_name]

    # Optional ensemble (stacking)
    if cfg["model"]["ensemble"]["enabled"] and len(candidates) >= 2:
        base_models = [
            {"name": name, "trainer": lambda X, y, c=candidates[name]: c["model"]}
            for name in candidates
        ]
        # For simplicity here, we skip retraining trainers; use selected best single model instead.
        logger.warning("Ensemble enabled in config but simple trainers are not provided; using best single model.")

    # Calibration
    calibrator = None
    if cfg["model"]["calibration"]["enabled"]:
        method = cfg["model"]["calibration"]["method"].lower()
        yva_proba_best = best["predict"](Xva)
        if method == "platt":
            calibrator = PlattCalibrator().fit(yva_proba_best, yva.values)
        elif method == "isotonic":
            calibrator = IsotonicCalibrator().fit(yva_proba_best, yva.values)
        else:
            logger.warning(f"Unknown calibration method: {method}; skipping calibration")

    # Save artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    art_dir = Path(cfg["paths"]["artifacts_dir"]) / ts
    art_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"features": Xtr.columns.tolist()}, art_dir / "features.pkl")
    joblib.dump(imputer, art_dir / "imputer.pkl")
    if best.get("scaler") is not None:
        joblib.dump(best["scaler"], art_dir / "scaler.pkl")
    joblib.dump(best["model"], art_dir / f"model_{best_name}.pkl")
    if calibrator is not None:
        joblib.dump(calibrator, art_dir / "calibrator.pkl")

    # Save metadata
    meta = {
        "best_model": best_name,
        "val_metrics": best["val"],
        "config": cfg,
        "timestamp": ts,
    }
    joblib.dump(meta, art_dir / "meta.pkl")

    logger.info(f"Saved artifacts to {art_dir}")


if __name__ == "__main__":
    main()

