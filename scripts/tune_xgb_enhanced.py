"""
Random-search XGBoost tuning on enriched gold features (winner task).
"""

from __future__ import annotations

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
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
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(rng.integers(3, 10)),
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.07])),
        "subsample": float(rng.uniform(0.6, 0.95)),
        "colsample_bytree": float(rng.uniform(0.5, 0.9)),
        "reg_alpha": float(rng.uniform(0.0, 2.5)),
        "reg_lambda": float(rng.uniform(0.0, 2.5)),
        "min_child_weight": float(rng.uniform(1.0, 10.0)),
        "gamma": float(rng.uniform(0.0, 2.0)),
        "tree_method": "hist",
        "random_state": 42,
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


def cv_logloss(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, params: Dict, num_boost_round: int = 600) -> float:
    tscv = TimeSeriesSplit(n_splits=5)
    losses = []
    for train_idx, val_idx in tscv.split(meta):
        dtr = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
        dva = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
        model = xgb.train(params, dtr, num_boost_round=num_boost_round, verbose_eval=False)
        proba = model.predict(dva)
        losses.append(log_loss(y.iloc[val_idx], proba))
    return float(np.mean(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random search XGBoost on enriched features.")
    parser.add_argument("--trials", type=int, default=20, help="Number of random trials.")
    parser.add_argument("--cv-rounds", type=int, default=700, help="Boost rounds during CV.")
    parser.add_argument("--boost-rounds", type=int, default=900, help="Boost rounds for final training.")
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
    logger.info(f"Starting XGBoost enhanced tuning with {n_iter} random trials")
    for i in range(1, n_iter + 1):
        params = sample_params(rng)
        loss = cv_logloss(X_imp, y, meta, params, num_boost_round=args.cv_rounds)
        logger.info(f"Trial {i}/{n_iter}: logloss={loss:.4f} params={params}")
        if loss < best["cv_logloss"]:
            best = {"params": params, "cv_logloss": loss}

    logger.info(f"Best CV logloss: {best['cv_logloss']:.4f}")

    dtrain = xgb.DMatrix(X_imp, label=y)
    model = xgb.train(best["params"], dtrain, num_boost_round=args.boost_rounds)
    proba = model.predict(dtrain)
    pred = (proba >= 0.5).astype(int)
    metrics = MetricsCalculator().classification(y.values, pred, proba)
    logger.info(f"Training metrics with best params: {metrics}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    art_dir = Path(cfg["paths"]["artifacts_dir"]) / f"{ts}_xgb_enhanced_tuned"
    art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": X_imp.columns.tolist()}, art_dir / "features.pkl")
    joblib.dump(imputer, art_dir / "imputer.pkl")
    joblib.dump(model, art_dir / "model_xgboost_tuned.pkl")
    meta_out = {
        "best_model": "xgboost_tuned",
        "cv_logloss": best["cv_logloss"],
        "train_metrics": metrics,
        "params": best["params"],
        "timestamp": ts,
        "config": cfg,
    }
    joblib.dump(meta_out, art_dir / "meta.pkl")
    (art_dir / "best_params.json").write_text(json.dumps(best["params"], indent=2))
    logger.info(f"Saved tuned XGBoost enhanced artifacts to {art_dir}")


if __name__ == "__main__":
    main()
