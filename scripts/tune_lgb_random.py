"""
Random-search LightGBM tuning with time-series CV on training split.
Retrains best model and saves artifacts with tuned params.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
from loguru import logger
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
import joblib

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.loaders import UFCDataLoader
from src.data.splitters import TemporalSplitter
from src.data.preprocessing import FeatureTypeImputationStrategy
from src.evaluation.metrics import MetricsCalculator


def sample_params(rng: np.random.Generator) -> dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": float(rng.choice([0.03, 0.05, 0.07, 0.1])),
        "num_leaves": int(rng.integers(15, 63)),
        "max_depth": int(rng.integers(3, 9)),
        "feature_fraction": float(rng.uniform(0.6, 0.9)),
        "bagging_fraction": float(rng.uniform(0.6, 0.9)),
        "bagging_freq": int(rng.integers(1, 8)),
        "min_data_in_leaf": int(rng.integers(10, 200)),
        "lambda_l1": float(rng.uniform(0.0, 2.0)),
        "lambda_l2": float(rng.uniform(0.0, 2.0)),
        "verbose": -1,
    }


def cv_logloss(X, y, params, n_splits=5, num_boost_round=300):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses = []
    for tr, va in tscv.split(X):
        train = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        valid = lgb.Dataset(X.iloc[va], label=y.iloc[va])
        model = lgb.train(params, train, num_boost_round=num_boost_round, valid_sets=[valid],
                          callbacks=[lgb.log_evaluation(period=0)],)
        proba = model.predict(X.iloc[va])
        losses.append(log_loss(y.iloc[va], proba))
    return float(np.mean(losses))


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

    # For speed, disable indicators during tuning
    imputer = FeatureTypeImputationStrategy(create_indicators=False).fit(Xtr_raw)
    Xtr = imputer.transform(Xtr_raw)
    Xva = imputer.transform(Xva_raw)

    rng = np.random.default_rng(cfg["training"]["random_state"])

    best = {"params": None, "cv_logloss": 1e9}
    n_iter = 20
    for i in range(1, n_iter + 1):
        params = sample_params(rng)
        loss = cv_logloss(Xtr, ytr, params, n_splits=5, num_boost_round=250)
        logger.info(f"Trial {i}/{n_iter}: logloss={loss:.4f} params={params}")
        if loss < best["cv_logloss"]:
            best = {"params": params, "cv_logloss": loss}

    logger.info(f"Best CV logloss: {best['cv_logloss']:.4f}")

    # Retrain on full train and evaluate on val
    train_data = lgb.Dataset(Xtr, label=ytr)
    model = lgb.train(best["params"], train_data, num_boost_round=400, callbacks=[lgb.log_evaluation(period=0)])
    proba = model.predict(Xva)
    pred = (proba >= 0.5).astype(int)

    metrics = MetricsCalculator()
    val_metrics = metrics.classification(yva.values, pred, proba)
    logger.info(f"Validation metrics: {val_metrics}")

    # Save artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    art_dir = Path(cfg["paths"]["artifacts_dir"]) / f"{ts}_lgb_tuned"
    art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": Xtr.columns.tolist()}, art_dir / "features.pkl")
    joblib.dump(imputer, art_dir / "imputer.pkl")
    joblib.dump(model, art_dir / "model_lightgbm_tuned.pkl")
    meta = {
        "best_model": "lightgbm_tuned",
        "val_metrics": val_metrics,
        "cv_logloss": best["cv_logloss"],
        "params": best["params"],
        "timestamp": ts,
        "config": cfg,
    }
    joblib.dump(meta, art_dir / "meta.pkl")
    (art_dir / "best_params.json").write_text(json.dumps(best["params"], indent=2))
    logger.info(f"Saved tuned artifacts to {art_dir}")


if __name__ == "__main__":
    main()

