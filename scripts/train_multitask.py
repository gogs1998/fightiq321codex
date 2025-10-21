"""
Train multi-task models for Winner (binary), Method (multiclass), and Round (ordinal/multiclass) using gold features.

Labels are sourced from FightIQ golden dataset by join on fight_url.
This script trains separate baseline models per task (for simplicity), saves artifacts, and reports validation metrics.

Usage:
  python fightiq_codex/scripts/train_multitask.py
"""

import sys
import argparse
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score
import lightgbm as lgb
import joblib
import json

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.models.calibration_multiclass import WinnerPlatt, TemperatureScaling
from src.data.loaders import UFCDataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-task models (winner/method/round).")
    parser.add_argument("--drop-features", type=str, default=None, help="Path to JSON file listing features to drop.")
    return parser.parse_args()


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


def _compute_sample_weights(y_codes: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_codes, minlength=n_classes)
    weights = np.ones_like(y_codes, dtype=np.float64)
    for cls_idx, count in enumerate(counts):
        if count > 0:
            weights[y_codes == cls_idx] = len(y_codes) / (n_classes * count)
    # Normalize to keep average weight ~1
    weights *= len(weights) / weights.sum()
    return weights


def _load_features_and_labels(cfg):
    gold = pd.read_parquet(ROOT / 'data' / 'gold_features.parquet')
    labels_path = _resolve_path(cfg["paths"]["golden_dataset"])
    lab = pd.read_csv(labels_path)
    for col in ['fight_url']:
        gold[col] = gold[col].astype(str).str.strip().str.rstrip('/')
        lab[col] = lab[col].astype(str).str.strip().str.rstrip('/')

    lab_cols = ['fight_url','winner_encoded','result','finish_round','event_date']
    lab_subset = lab[[c for c in lab_cols if c in lab.columns]].drop_duplicates(subset=['fight_url'])

    loader = UFCDataLoader(cfg["paths"]["data_dir"])
    feature_cols = loader.get_feature_columns(lab, exclude_odds=cfg["features"]["exclude_odds"])
    orig_feat = lab[['fight_url'] + feature_cols].copy()
    orig_feat = orig_feat.groupby('fight_url', as_index=False).first()

    df = gold.merge(lab_subset, on='fight_url', how='inner', suffixes=('_gold','_lab'))
    df = df.merge(orig_feat, on='fight_url', how='left')

    if 'event_date_gold' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date_gold'], errors='coerce')
    elif 'event_date_lab' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date_lab'], errors='coerce')
    else:
        df['event_date'] = pd.NaT

    y_winner = df['winner_encoded'] if 'winner_encoded' in df.columns else None
    y_method = None
    if 'result' in df.columns:
        def map_method(x: str):
            x = str(x).lower()
            if any(k in x for k in ['ko','tko']):
                return 'KO_TKO'
            if 'sub' in x:
                return 'SUB'
            if 'dec' in x or 'decision' in x:
                return 'DEC'
            return 'OTHER'
        y_method = df['result'].map(map_method)
    y_round = df['finish_round'] if 'finish_round' in df.columns else None

    drop_cols = [c for c in ['fight_url','event_url','event_name','event_date','winner_encoded','result','finish_round'] if c in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number]).fillna(0)
    return df[['fight_url','event_date']], X, y_winner, y_method, y_round


def _time_cv_split(meta: pd.DataFrame, n_splits=5):
    # Assumes event_date is present for temporal ordering
    order = meta.sort_values('event_date').reset_index(drop=True)
    effective = max(2, min(n_splits, len(order) - 1))
    tscv = TimeSeriesSplit(n_splits=effective)
    for tr_idx, va_idx in tscv.split(order):
        tr_ids = order.loc[tr_idx, 'fight_url']
        va_ids = order.loc[va_idx, 'fight_url']
        yield tr_ids.values, va_ids.values


def train_task(meta, X, y, task_name: str):
    if y is None:
        logger.warning(f"Task {task_name}: labels missing; skipping")
        return None

    mask = pd.notna(y)
    if not mask.any():
        logger.warning(f"Task {task_name}: no non-null labels; skipping")
        return None

    Xf = X[mask].reset_index(drop=True)
    yf = pd.Series(y[mask]).reset_index(drop=True)
    mf = meta[mask].reset_index(drop=True)

    if len(Xf) < 100:
        logger.warning(f"Task {task_name}: insufficient rows ({len(Xf)}); skipping")
        return None

    max_rounds = 800
    params = {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
    }

    if task_name == "winner":
        classes = [0, 1]
        params.update({"objective": "binary", "metric": "binary_logloss"})
        y_codes = yf.astype(int).to_numpy()
        calibration_input = "proba"
    else:
        if task_name == "method":
            classes = sorted(yf.unique().tolist())
        else:
            classes = sorted(pd.to_numeric(yf.dropna()).unique().tolist())
        params.update({"objective": "multiclass", "metric": "multi_logloss", "num_class": len(classes)})
        cat = pd.Categorical(yf, categories=classes)
        y_codes = cat.codes.astype(int)
        calibration_input = "logits"

    if np.any(y_codes < 0):
        valid_mask = y_codes >= 0
        Xf = Xf.loc[valid_mask].reset_index(drop=True)
        y_codes = y_codes[valid_mask]
        yf = yf.loc[valid_mask].reset_index(drop=True)
        mf = mf.loc[valid_mask].reset_index(drop=True)

    sample_weights = _compute_sample_weights(y_codes, len(classes))
    class_counts = np.bincount(y_codes, minlength=len(classes))
    class_weights = {
        classes[idx]: float(len(y_codes) / (len(classes) * count))
        for idx, count in enumerate(class_counts)
        if count > 0
    }

    scores = []
    models = []
    best_iterations = []
    effective_splits = max(2, min(5, len(Xf) - 1))
    for tr_ids, va_ids in _time_cv_split(mf[["fight_url", "event_date"]], n_splits=effective_splits):
        tr_mask = mf["fight_url"].isin(tr_ids).to_numpy()
        va_mask = mf["fight_url"].isin(va_ids).to_numpy()
        dtr = lgb.Dataset(
            Xf.loc[tr_mask],
            label=y_codes[tr_mask],
            weight=sample_weights[tr_mask],
            free_raw_data=False,
        )
        dva = lgb.Dataset(
            Xf.loc[va_mask],
            label=y_codes[va_mask],
            weight=sample_weights[va_mask],
            free_raw_data=False,
        )
        mdl = lgb.train(
            params,
            dtr,
            num_boost_round=max_rounds,
            valid_sets=[dva],
            valid_names=["val"],
            callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(50, verbose=False)],
        )
        best_iter = mdl.best_iteration or max_rounds
        best_iterations.append(best_iter)
        if task_name == "winner":
            proba = mdl.predict(Xf.loc[va_mask], num_iteration=best_iter)
            scores.append(
                {
                    "logloss": float(log_loss(y_codes[va_mask], proba, labels=[0, 1])),
                    "acc": float(accuracy_score(y_codes[va_mask], (proba >= 0.5).astype(int))),
                    "best_iteration": best_iter,
                }
            )
        else:
            proba = mdl.predict(Xf.loc[va_mask], num_iteration=best_iter)
            ytrue = y_codes[va_mask]
            scores.append(
                {
                    "logloss": float(log_loss(ytrue, proba, labels=list(range(len(classes))))),
                    "acc": float(accuracy_score(ytrue, proba.argmax(axis=1))),
                    "best_iteration": best_iter,
                }
            )
        models.append(mdl)

    logger.info(f"Task {task_name} CV scores: {scores}")

    order = mf.sort_values("event_date").reset_index(drop=True)
    split_idx = int(len(order) * 0.9)
    cal_ids = order.loc[split_idx:, "fight_url"]
    cal_mask = mf["fight_url"].isin(cal_ids).to_numpy()
    tr_mask_all = ~cal_mask

    dtrain = lgb.Dataset(
        Xf.loc[tr_mask_all],
        label=y_codes[tr_mask_all],
        weight=sample_weights[tr_mask_all],
        free_raw_data=False,
    )
    dcal = lgb.Dataset(
        Xf.loc[cal_mask],
        label=y_codes[cal_mask],
        weight=sample_weights[cal_mask],
        free_raw_data=False,
    )
    final = lgb.train(
        params,
        dtrain,
        num_boost_round=max_rounds,
        valid_sets=[dcal],
        valid_names=["cal"],
        callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(50, verbose=False)],
    )
    best_iter = final.best_iteration or max_rounds

    if task_name == "winner":
        proba_cal = final.predict(Xf.loc[cal_mask], num_iteration=best_iter)
        calibrator = WinnerPlatt().fit(proba_cal, yf.loc[cal_mask].to_numpy())
    else:
        logits = final.predict(Xf.loc[cal_mask], num_iteration=best_iter, raw_score=True)
        calibrator = TemperatureScaling().fit(logits, y_codes[cal_mask])

    return {
        "model": final,
        "classes": classes,
        "cv_scores": scores,
        "calibrator": calibrator,
        "best_iteration": int(best_iter),
        "class_weights": class_weights,
        "calibration_input": calibration_input,
    }


def _aggregate_scores(scores: list[dict]) -> dict:
    if not scores:
        return {}
    df = pd.DataFrame(scores)
    result = {
        "mean_logloss": float(df['logloss'].mean()),
        "mean_accuracy": float(df['acc'].mean()),
        "folds": len(scores),
    }
    if "best_iteration" in df.columns:
        result["mean_best_iteration"] = float(df["best_iteration"].mean())
    return result


def main():
    args = parse_args()
    cfg = load_config(ROOT / 'config/config.yaml')
    logger.remove()
    logger.add(sys.stderr, level=cfg.get('logging',{}).get('level','INFO'))

    meta, X, y_win, y_met, y_rnd = _load_features_and_labels(cfg)
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
        X = X.drop(columns=list(drop_features), errors='ignore')
    out_dir = ROOT / 'artifacts' / 'multitask'
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_list = X.columns.tolist()
    joblib.dump({"features": feature_list}, out_dir / "features.pkl")

    summary_payload = {
        "meta": {
            "drop_features_count": len(drop_features),
            "drop_source": str(drop_source) if drop_source else None,
        }
    }

    # Winner
    res_win = train_task(meta, X, y_win, 'winner')
    if res_win:
        joblib.dump(res_win, out_dir / 'winner_lgb.pkl')
        summary_payload['winner'] = {
            **_aggregate_scores(res_win.get('cv_scores', [])),
            "best_iteration": res_win.get("best_iteration"),
            "calibration_input": res_win.get("calibration_input"),
            "class_weights": res_win.get("class_weights"),
        }

    # Method
    res_met = train_task(meta, X, y_met, 'method')
    if res_met:
        joblib.dump(res_met, out_dir / 'method_lgb.pkl')
        summary_payload['method'] = {
            **_aggregate_scores(res_met.get('cv_scores', [])),
            "best_iteration": res_met.get("best_iteration"),
            "classes": res_met.get("classes"),
            "calibration_input": res_met.get("calibration_input"),
            "class_weights": res_met.get("class_weights"),
        }

    # Round
    res_rnd = train_task(meta, X, y_rnd, 'round')
    if res_rnd:
        joblib.dump(res_rnd, out_dir / 'round_lgb.pkl')
        summary_payload['round'] = {
            **_aggregate_scores(res_rnd.get('cv_scores', [])),
            "best_iteration": res_rnd.get("best_iteration"),
            "classes": res_rnd.get("classes"),
            "calibration_input": res_rnd.get("calibration_input"),
            "class_weights": res_rnd.get("class_weights"),
        }

    logger.info(f"Saved multi-task artifacts to {out_dir}")
    summary_path = ROOT / 'outputs' / 'multitask_train_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    logger.info(f"Wrote multi-task summary to {summary_path}")


if __name__ == '__main__':
    main()
