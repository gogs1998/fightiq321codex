"""
Random-search LightGBM tuning on enriched gold features with original leak-safe features merged in.
Uses time-series CV, retrains the best model, and saves tuned params.
"""

from __future__ import annotations

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).parents[1]

import sys

sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.preprocessing import FeatureTypeImputationStrategy
from src.evaluation.metrics import MetricsCalculator
from scripts.train_winner_enhanced import _load_gold_with_labels  # type: ignore


def sample_params(rng: np.random.Generator) -> Dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.07])),
        "num_leaves": int(rng.integers(24, 80)),
        "max_depth": int(rng.integers(3, 9)),
        "feature_fraction": float(rng.uniform(0.5, 0.9)),
        "bagging_fraction": float(rng.uniform(0.6, 0.95)),
        "bagging_freq": int(rng.integers(1, 8)),
        "min_data_in_leaf": int(rng.integers(20, 400)),
        "lambda_l1": float(rng.uniform(0.0, 2.5)),
        "lambda_l2": float(rng.uniform(0.0, 2.5)),
        "verbose": -1,
    }


def prepare_data(cfg):
    df = _load_gold_with_labels(cfg)
    mask = pd.notna(df["winner_encoded"]) & df["winner_encoded"].isin([0, 1])
    df = df[mask].copy()
    drop_cols = [
        c
        for c in [
            "fight_url",
            "event_url",
            "event_name",
            "event_date",
            "winner_encoded",
            "result",
            "finish_round",
        ]
        if c in df.columns
    ]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    y = df["winner_encoded"].astype(int)
    meta = df[["fight_url", "event_date"]]
    return meta, X, y


def cv_logloss(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, params: Dict, num_boost_round: int = 400) -> float:
    tscv = TimeSeriesSplit(n_splits=5)
    losses = []
    for train_idx, val_idx in tscv.split(meta):
        Xtr, Xva = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yva = y.iloc[train_idx], y.iloc[val_idx]
        dtr = lgb.Dataset(Xtr, label=ytr)
        model = lgb.train(params, dtr, num_boost_round=num_boost_round, callbacks=[lgb.log_evaluation(period=0)])
        proba = model.predict(Xva)
        losses.append(log_loss(yva, proba))
    return float(np.mean(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random search LightGBM on enriched features.")
    parser.add_argument("--trials", type=int, default=20, help="Number of random trials.")
    parser.add_argument("--boost-rounds", type=int, default=600, help="Boosting rounds for final model.")
    parser.add_argument("--cv-rounds", type=int, default=500, help="Boosting rounds during CV.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(ROOT / "config" / "config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    meta, X, y = prepare_data(cfg)

    imputer = FeatureTypeImputationStrategy(create_indicators=True).fit(X)
    X_imp = imputer.transform(X)

    rng = np.random.default_rng(cfg["training"]["random_state"])
    best = {"params": None, "cv_logloss": 1e9}
    n_iter = args.trials
    logger.info(f"Starting LightGBM enhanced tuning with {n_iter} random trials")
    for i in range(1, n_iter + 1):
        params = sample_params(rng)
        loss = cv_logloss(X_imp, y, meta, params, num_boost_round=args.cv_rounds)
        logger.info(f"Trial {i}/{n_iter}: logloss={loss:.4f} params={params}")
        if loss < best["cv_logloss"]:
            best = {"params": params, "cv_logloss": loss}

    logger.info(f"Best CV logloss: {best['cv_logloss']:.4f}")

    train_data = lgb.Dataset(X_imp, label=y)
    model = lgb.train(best["params"], train_data, num_boost_round=args.boost_rounds, callbacks=[lgb.log_evaluation(period=0)])
    proba = model.predict(X_imp)
    pred = (proba >= 0.5).astype(int)
    metrics = MetricsCalculator().classification(y.values, pred, proba)
    logger.info(f"Training metrics with best params: {metrics}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    art_dir = Path(cfg["paths"]["artifacts_dir"]) / f"{ts}_lgb_enhanced_tuned"
    art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": X_imp.columns.tolist()}, art_dir / "features.pkl")
    joblib.dump(imputer, art_dir / "imputer.pkl")
    joblib.dump(model, art_dir / "model_lightgbm_tuned.pkl")
    meta_out = {
        "best_model": "lightgbm_tuned",
        "cv_logloss": best["cv_logloss"],
        "train_metrics": metrics,
        "params": best["params"],
        "timestamp": ts,
        "config": cfg,
    }
    joblib.dump(meta_out, art_dir / "meta.pkl")
    (art_dir / "best_params.json").write_text(json.dumps(best["params"], indent=2))
    logger.info(f"Saved tuned LightGBM enhanced artifacts to {art_dir}")


if __name__ == "__main__":
    main()
