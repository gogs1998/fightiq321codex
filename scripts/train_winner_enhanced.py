"""
Train tuned winner models on enriched gold features and build a simple stacking ensemble.

Workflow:
 1. Load enhanced gold features (built via scripts/build_gold_features.py).
 2. Merge with original golden labels (winner_encoded).
 3. Apply temporal train/val/test split using config thresholds.
 4. Fit FeatureTypeImputationStrategy on train, transform val/test.
 5. Train tuned LightGBM and XGBoost models (params sourced from artifacts).
 6. Fit Platt calibrators on validation predictions.
 7. Train a logistic stacking model on validation probabilities (LGB + XGB).
 8. Evaluate val/test metrics for each model (raw + calibrated).
 9. Persist artifacts to artifacts/winner_enhanced/<timestamp>.
10. Export metrics to outputs/winner_enhanced_summary.csv and per-model JSON metadata.
"""

from __future__ import annotations

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

ROOT = Path(__file__).parents[1]

import sys

sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.splitters import TemporalSplitter
from src.data.preprocessing import FeatureTypeImputationStrategy
from src.data.validation import DataValidator
from src.evaluation.metrics import MetricsCalculator
from src.models.calibration import PlattCalibrator
from src.data.loaders import UFCDataLoader


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    for base in [ROOT, ROOT.parent]:
        candidate = base / path
        if candidate.exists():
            return candidate
    return ROOT / path


def _find_latest_zero_importance() -> Path | None:
    out_root = ROOT / "outputs"
    if not out_root.exists():
        return None
    candidates = []
    for pattern in ["winner_zero_importance_features_*.json", "winner_zero_importance_features.json"]:
        candidates.extend(out_root.glob(pattern))
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except OSError:
        return None


def _load_params(path: str | Path | None) -> Dict:
    if not path:
        return {}
    p = _resolve_path(path)
    if not p.exists():
        logger.warning(f"Tuned params not found at {p}; using default hyper-parameters.")
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_gold_with_labels(cfg: Dict) -> pd.DataFrame:
    gold_path = ROOT / "data" / "gold_features.parquet"
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold features not found at {gold_path}. Run build_gold_features.py first.")
    gold = pd.read_parquet(gold_path)

    labels_path = _resolve_path(cfg["paths"]["golden_dataset"])
    golden = pd.read_csv(labels_path)

    for col in ["fight_url"]:
        gold[col] = gold[col].astype(str).str.strip().str.rstrip("/")
        golden[col] = golden[col].astype(str).str.strip().str.rstrip("/")

    # Targets + minimal metadata
    target_cols = ["winner", "winner_encoded", "result", "result_details", "finish_round", "finish_time"]
    meta_cols = [
        "event_date",
        "event_name",
        "event_url",
        "event_city",
        "event_state",
        "event_country",
        "f_1_name",
        "f_2_name",
        "f_1_url",
        "f_2_url",
    ]
    keep_cols = ["fight_url"] + [c for c in target_cols + ["event_date"] if c in golden.columns]
    labels = golden[keep_cols].drop_duplicates(subset=["fight_url"])

    # Leak-safe original feature block using loader filters
    loader = UFCDataLoader(cfg["paths"]["data_dir"])
    feature_cols = loader.get_feature_columns(golden, exclude_odds=cfg["features"]["exclude_odds"])
    orig_feat = golden[["fight_url"] + feature_cols].copy()
    orig_feat = orig_feat.groupby("fight_url", as_index=False).first()

    merged = gold.merge(labels, on="fight_url", how="inner", suffixes=("_gold", "_lab"))
    merged = merged.merge(orig_feat, on="fight_url", how="left")

    event_cols = [c for c in ["event_date_gold", "event_date_lab", "event_date"] if c in merged.columns]
    merged["event_date"] = pd.NaT
    for c in event_cols:
        merged["event_date"] = merged["event_date"].fillna(pd.to_datetime(merged[c], errors="coerce"))

    missing_dates = merged["event_date"].isna().sum()
    if missing_dates > 0:
        logger.warning(f"{missing_dates} rows missing event_date after merge; dropping.")
        merged = merged.dropna(subset=["event_date"])

    merged = merged.sort_values("event_date").reset_index(drop=True)
    return merged


def _load_parity_bundle(cfg: Dict):
    par_root = ROOT / "artifacts" / "parity_winner"
    if not par_root.exists():
        return None
    candidates = sorted([p for p in par_root.iterdir() if p.is_dir()])
    if not candidates:
        return None
    par_dir = candidates[-1]
    try:
        imputer = joblib.load(par_dir / "imputer.pkl")
        features = joblib.load(par_dir / "features.pkl")["features"]
        meta = joblib.load(par_dir / "lightgbm_meta.pkl")
        best_iter = meta.get("best_iteration")
        booster = lgb.Booster(model_file=str(par_dir / "lightgbm_model.txt"))
        calibrator = None
        cal_path = par_dir / "lightgbm_calibrator.pkl"
        if cal_path.exists():
            calibrator = joblib.load(cal_path)
        logger.info(f"Loaded parity bundle from {par_dir} with {len(features)} features")
        return {
            "booster": booster,
            "imputer": imputer,
            "features": features,
            "calibrator": calibrator,
            "best_iter": best_iter,
        }
    except Exception as exc:
        logger.warning(f"Failed loading parity bundle from {par_dir}: {exc}")
        return None


def _compute_parity_probabilities(
    cfg: Dict,
    fight_urls: pd.Series,
    loader: UFCDataLoader,
    bundle,
) -> pd.Series:
    booster = bundle["booster"]
    imputer: FeatureTypeImputationStrategy = bundle["imputer"]
    features = bundle["features"]
    calibrator = bundle["calibrator"]
    best_iter = bundle["best_iter"]

    golden_path = _resolve_path(cfg["paths"]["golden_dataset"])
    golden = loader.load_golden_dataset(str(golden_path))
    golden["fight_url"] = golden["fight_url"].astype(str).str.strip().str.rstrip("/")
    mask = golden["fight_url"].isin(fight_urls.unique())
    if not mask.any():
        return pd.Series(dtype=float, name="parity_prob")
    golden = golden[mask].copy()
    X_raw, _ = loader.prepare_features_target(
        golden,
        target="winner_encoded",
        exclude_odds=cfg["features"]["exclude_odds"],
        remove_draws=False,
    )
    X_raw["fight_url"] = golden["fight_url"].values
    grouped = X_raw.groupby("fight_url").first()
    base_features = [f for f in features if not f.endswith("_missing")]
    aligned = loader.align_to_training_features(grouped.copy(), base_features)
    X_imp = imputer.transform(aligned)
    if isinstance(X_imp, pd.DataFrame):
        X_imp_df = X_imp
    else:
        X_imp_df = pd.DataFrame(X_imp, columns=aligned.columns, index=aligned.index)
    X_imp_df = X_imp_df[bundle["features"]]

    pred = booster.predict(X_imp_df, num_iteration=best_iter)
    if calibrator is not None:
        pred = calibrator.transform(pred)
    return pd.Series(pred, index=X_imp_df.index, name="parity_prob")


def _prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "winner_encoded" not in df:
        raise ValueError("winner_encoded column missing from merged dataset.")
    mask = df["winner_encoded"].isin([0, 1])
    dropped = int((~mask).sum())
    if dropped:
        logger.warning(f"Dropping {dropped} rows with non-binary winner labels.")
    df_clean = df.loc[mask].copy()
    feature_cols = [c for c in numeric_cols if c != "winner_encoded"]
    X = df_clean[feature_cols].copy()
    y = df_clean["winner_encoded"].astype(int)
    return X, y


def _train_lightgbm(params: Dict, Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series):
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
    callbacks = [lgb.log_evaluation(period=0), lgb.early_stopping(100, verbose=False)]
    booster = lgb.train(
        base_params,
        train_set,
        num_boost_round=3000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    best_iter = booster.best_iteration or booster.current_iteration()
    return booster, int(best_iter)


def _train_xgboost(params: Dict, Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series):
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
        num_boost_round=3000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    best_iter = booster.best_iteration or booster.num_boosted_rounds()
    return booster, int(best_iter)


def _compute_metrics(
    metrics: MetricsCalculator,
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    calibrated: bool = False,
) -> Dict[str, float | str]:
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


def _compute_threshold_metrics(
    metrics: MetricsCalculator,
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Dict[str, float | str]:
    y_pred = (y_proba >= threshold).astype(int)
    cls = metrics.classification(y_true, y_pred, y_proba)
    cal = metrics.calibration(y_true, y_proba)
    row: Dict[str, float | str] = {
        "model": model_name,
        "split": split_name,
        "calibrated": f"thr={threshold:.3f}",
    }
    row.update({k: float(v) for k, v in cls.items()})
    row.update(cal)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train winner models on enriched gold features.")
    parser.add_argument("--min-val-acc", type=float, default=0.0, help="Minimum validation accuracy guard for the selected model.")
    parser.add_argument("--min-test-acc", type=float, default=0.0, help="Minimum test accuracy guard for the selected model.")
    parser.add_argument("--guard-model", type=str, default="stack_ensemble", help="Model name to apply accuracy guards (default: stack_ensemble).")
    parser.add_argument("--drop-features", type=str, default=None, help="Path to JSON file listing features to drop before training.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(ROOT / "config" / "config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    drop_features: set[str] = set()
    modeling_cfg = cfg.get("modeling", {})
    drop_source = args.drop_features or modeling_cfg.get("drop_features_path")
    if not drop_source and modeling_cfg.get("auto_use_latest_zero_importance", False):
        auto_path = _find_latest_zero_importance()
        if auto_path:
            drop_source = str(auto_path)
            logger.info(f"Auto-loading latest zero-importance feature list from {auto_path}.")
        else:
            logger.info("Configured to auto-use zero-importance features, but none found under outputs/.")
    if drop_source:
        drop_path = _resolve_path(drop_source)
        drop_features = set(json.loads(Path(drop_path).read_text()))
        logger.info(f"Dropping {len(drop_features)} features specified in {drop_path}.")

    merged = _load_gold_with_labels(cfg)
    parity_bundle = _load_parity_bundle(cfg)
    parity_series = None
    if parity_bundle:
        loader_for_parity = UFCDataLoader(cfg["paths"]["data_dir"])
        parity_series = _compute_parity_probabilities(cfg, merged["fight_url"], loader_for_parity, parity_bundle)
    else:
        logger.warning("Parity winner artifacts not found; stacking will not use parity signal.")
    validator = DataValidator(merged)
    validation_result = validator.validate_all()
    if not validation_result["passed"]:
        for err in validation_result["errors"]:
            logger.error(err)
        raise RuntimeError("Data validation failed; aborting training.")
    for warn in validation_result["warnings"]:
        logger.warning(warn)

    splitter = TemporalSplitter(cfg["splits"]["val_start_date"], cfg["splits"]["test_start_date"])
    split = splitter.split(merged)

    mask_train = split.train["winner_encoded"].isin([0, 1])
    mask_val = split.val["winner_encoded"].isin([0, 1])
    mask_test = split.test["winner_encoded"].isin([0, 1])
    if parity_series is not None:
        parity_train = split.train.loc[mask_train, "fight_url"].map(parity_series).fillna(0.5).to_numpy()
        parity_val = split.val.loc[mask_val, "fight_url"].map(parity_series).fillna(0.5).to_numpy()
        parity_test = split.test.loc[mask_test, "fight_url"].map(parity_series).fillna(0.5).to_numpy()
    else:
        parity_train = np.full(mask_train.sum(), 0.5)
        parity_val = np.full(mask_val.sum(), 0.5)
        parity_test = np.full(mask_test.sum(), 0.5)

    Xtr_raw, ytr = _prepare_features_target(split.train)
    Xva_raw, yva = _prepare_features_target(split.val)
    Xte_raw, yte = _prepare_features_target(split.test)

    if drop_features:
        Xtr_raw = Xtr_raw.drop(columns=list(drop_features), errors="ignore")
        Xva_raw = Xva_raw.drop(columns=list(drop_features), errors="ignore")
        Xte_raw = Xte_raw.drop(columns=list(drop_features), errors="ignore")

    imputer = FeatureTypeImputationStrategy(create_indicators=cfg["features"]["create_missing_indicators"]).fit(Xtr_raw)
    Xtr = imputer.transform(Xtr_raw).astype(np.float32)
    Xva = imputer.transform(Xva_raw).astype(np.float32)
    Xte = imputer.transform(Xte_raw).astype(np.float32)

    tuned_lgb_params = _load_params(cfg.get("modeling", {}).get("tuned_lgb_params_path"))
    tuned_xgb_params = _load_params(cfg.get("modeling", {}).get("tuned_xgb_params_path"))

    logger.info("Training LightGBM with enriched gold features...")
    lgb_model, lgb_best_iter = _train_lightgbm(tuned_lgb_params, Xtr, ytr, Xva, yva)
    proba_val_lgb = lgb_model.predict(Xva, num_iteration=lgb_best_iter)
    proba_test_lgb = lgb_model.predict(Xte, num_iteration=lgb_best_iter)
    lgb_cal = PlattCalibrator().fit(proba_val_lgb, yva.values)
    proba_test_lgb_cal = lgb_cal.transform(proba_test_lgb)

    logger.info("Training XGBoost with enriched gold features...")
    xgb_model, xgb_best_iter = _train_xgboost(tuned_xgb_params, Xtr, ytr, Xva, yva)
    dval = xgb.DMatrix(Xva)
    dtest = xgb.DMatrix(Xte)
    proba_val_xgb = xgb_model.predict(dval, iteration_range=(0, xgb_best_iter))
    proba_test_xgb = xgb_model.predict(dtest, iteration_range=(0, xgb_best_iter))
    xgb_cal = PlattCalibrator().fit(proba_val_xgb, yva.values)
    proba_test_xgb_cal = xgb_cal.transform(proba_test_xgb)

    logger.info("Training logistic stacking ensemble on validation probabilities...")
    stack_input_val = np.vstack([proba_val_lgb, proba_val_xgb, parity_val]).T
    stack_input_test = np.vstack([proba_test_lgb, proba_test_xgb, parity_test]).T
    stacker = LogisticRegression(max_iter=1000)
    stacker.fit(stack_input_val, yva.values)
    proba_val_stack = stacker.predict_proba(stack_input_val)[:, 1]
    proba_test_stack = stacker.predict_proba(stack_input_test)[:, 1]

    metrics = MetricsCalculator()
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thresh = 0.5
    best_val_acc = accuracy_score(yva.values, (proba_val_stack >= best_thresh).astype(int))
    for thr in thresholds:
        acc = accuracy_score(yva.values, (proba_val_stack >= thr).astype(int))
        if acc > best_val_acc:
            best_val_acc = acc
            best_thresh = thr
    stack_val_threshold_metrics = _compute_threshold_metrics(metrics, "stack_ensemble_opt", "val", yva.values, proba_val_stack, best_thresh)
    stack_test_threshold_metrics = _compute_threshold_metrics(metrics, "stack_ensemble_opt", "test", yte.values, proba_test_stack, best_thresh)

    rows: List[Dict[str, float | str]] = []

    rows.append(_compute_metrics(metrics, "lightgbm_tuned", "val", yva.values, proba_val_lgb))
    rows.append(_compute_metrics(metrics, "lightgbm_tuned", "test", yte.values, proba_test_lgb))
    rows.append(_compute_metrics(metrics, "lightgbm_tuned", "test", yte.values, proba_test_lgb_cal, calibrated=True))

    rows.append(_compute_metrics(metrics, "xgboost_tuned", "val", yva.values, proba_val_xgb))
    rows.append(_compute_metrics(metrics, "xgboost_tuned", "test", yte.values, proba_test_xgb))
    rows.append(_compute_metrics(metrics, "xgboost_tuned", "test", yte.values, proba_test_xgb_cal, calibrated=True))

    rows.append(_compute_metrics(metrics, "stack_ensemble", "val", yva.values, proba_val_stack))
    rows.append(_compute_metrics(metrics, "stack_ensemble", "test", yte.values, proba_test_stack))
    rows.append(stack_val_threshold_metrics)
    rows.append(stack_test_threshold_metrics)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Feature importance analysis
    lgb_gain = pd.Series(lgb_model.feature_importance(importance_type="gain"), index=Xtr.columns, name="lgb_gain")
    xgb_gain_dict = xgb_model.get_score(importance_type="gain")
    xgb_gain = pd.Series([xgb_gain_dict.get(col, 0.0) for col in Xtr.columns], index=Xtr.columns, name="xgb_gain")
    importance_df = pd.concat([lgb_gain, xgb_gain], axis=1).fillna(0.0)
    importance_df.insert(0, "feature", importance_df.index)
    importance_path = ROOT / "outputs" / f"winner_feature_importance_{ts}.csv"
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_path, index=False)
    zero_features = importance_df[(importance_df["lgb_gain"] <= 0.0) & (importance_df["xgb_gain"] <= 0.0)]["feature"].tolist()
    zero_path = ROOT / "outputs" / f"winner_zero_importance_features_{ts}.json"
    zero_path.write_text(json.dumps(zero_features, indent=2))
    logger.info(f"Wrote feature importances to {importance_path} (zero-importance features: {len(zero_features)})")

    summary_df = pd.DataFrame(rows)
    outputs_path = ROOT / "outputs" / "winner_enhanced_summary.csv"
    summary_df.to_csv(outputs_path, index=False)
    logger.info(f"Saved metrics summary to {outputs_path}")

    guard_model = args.guard_model
    guard_val = summary_df[
        (summary_df["model"] == guard_model) & (summary_df["split"] == "val") & (summary_df["calibrated"] == "no")
    ]
    guard_test = summary_df[
        (summary_df["model"] == guard_model) & (summary_df["split"] == "test") & (summary_df["calibrated"] == "no")
    ]
    guard_val_acc = float(guard_val["accuracy"].iloc[0]) if not guard_val.empty else None
    guard_test_acc = float(guard_test["accuracy"].iloc[0]) if not guard_test.empty else None
    logger.info(
        f"Guard model {guard_model}: val_acc={guard_val_acc}, test_acc={guard_test_acc}, "
        f"thresholds=({args.min_val_acc}, {args.min_test_acc})"
    )
    if args.min_val_acc and (guard_val_acc is None or guard_val_acc < args.min_val_acc):
        logger.error(
            f"Validation accuracy {guard_val_acc} below required threshold {args.min_val_acc} for {guard_model}."
        )
        sys.exit(2)
    if args.min_test_acc and (guard_test_acc is None or guard_test_acc < args.min_test_acc):
        logger.error(
            f"Test accuracy {guard_test_acc} below required threshold {args.min_test_acc} for {guard_model}."
        )
        sys.exit(3)

    artifacts_root = ROOT / "artifacts" / "winner_enhanced"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    run_dir = artifacts_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"features": list(Xtr.columns)}, run_dir / "features.pkl")
    joblib.dump(imputer, run_dir / "imputer.pkl")
    lgb_model.save_model(str(run_dir / "lightgbm_model.txt"))
    joblib.dump({"params": tuned_lgb_params, "best_iteration": lgb_best_iter}, run_dir / "lightgbm_meta.pkl")
    joblib.dump(lgb_cal, run_dir / "lightgbm_calibrator.pkl")

    xgb_model.save_model(str(run_dir / "xgboost_model.json"))
    joblib.dump({"params": tuned_xgb_params, "best_iteration": xgb_best_iter}, run_dir / "xgboost_meta.pkl")
    joblib.dump(xgb_cal, run_dir / "xgboost_calibrator.pkl")

    joblib.dump(stacker, run_dir / "stacker.pkl")

    meta = {
        "timestamp": ts,
        "train_rows": int(len(Xtr)),
        "val_rows": int(len(Xva)),
        "test_rows": int(len(Xte)),
        "lightgbm_best_iteration": lgb_best_iter,
        "xgboost_best_iteration": xgb_best_iter,
        "summary_csv": str(outputs_path),
        "guard_model": guard_model,
        "guard_metrics": {
            "val_accuracy": guard_val_acc,
            "test_accuracy": guard_test_acc,
            "min_val_acc": args.min_val_acc,
            "min_test_acc": args.min_test_acc,
        },
        "stack_threshold": best_thresh,
        "stack_val_accuracy_threshold": float(best_val_acc),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
