"""
Predict upcoming fights using saved artifacts and robust alignment.
Outputs predictions and suggested Kelly bet sizes when odds present.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from loguru import logger
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.loaders import UFCDataLoader
from src.data.preprocessing import FeatureTypeImputationStrategy
from src.betting.kelly_criterion import kelly_fraction


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    for base in [ROOT, ROOT.parent]:
        candidate = base / path
        if candidate.exists():
            return candidate
    return ROOT / path


def _predict_with_model(model, name: str, X: pd.DataFrame) -> np.ndarray:
    lname = name.lower()
    if lname.startswith("xgb") or lname == "xgboost":
        return model.predict(xgb.DMatrix(X))
    if lname.startswith("lgb") or lname.startswith("lightgbm"):
        return model.predict(X)
    return model.predict_proba(X)[:, 1]


def _load_parity_bundle(cfg):
    par_root = ROOT / "artifacts" / "parity_winner"
    if not par_root.exists():
        return None
    dirs = sorted([p for p in par_root.iterdir() if p.is_dir()])
    if not dirs:
        return None
    par_dir = dirs[-1]
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
        return {
            "booster": booster,
            "imputer": imputer,
            "features": features,
            "calibrator": calibrator,
            "best_iter": best_iter,
        }
    except Exception as exc:
        logger.warning(f"Failed to load parity bundle: {exc}")
        return None


def _find_latest_winner_artifact(cfg) -> Path:
    raw_base = Path(cfg["paths"]["artifacts_dir"])
    candidate_paths = []
    for base in {raw_base, ROOT / raw_base, ROOT / "artifacts"}:
        if base.exists():
            candidate_paths.append(base)
        sub = base / "winner_enhanced"
        if sub.exists():
            candidate_paths.append(sub)
    run_dirs = []
    for root in candidate_paths:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if (child / "lightgbm_model.txt").exists() and (child / "xgboost_model.json").exists():
                run_dirs.append(child)
    if not run_dirs:
        raise FileNotFoundError(f"No trained winner artifacts found under {raw_base}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _compute_parity_probs(loader: UFCDataLoader, features_df: pd.DataFrame, fight_urls: pd.Series, bundle) -> pd.Series:
    feats = features_df.copy()
    feats.index = fight_urls
    aligned = loader.align_to_training_features(feats, bundle["features"])
    X_imp = bundle["imputer"].transform(aligned)
    if isinstance(X_imp, pd.DataFrame):
        X_imp_df = X_imp
    else:
        X_imp_df = pd.DataFrame(X_imp, columns=aligned.columns, index=aligned.index)
    X_imp_df = X_imp_df[bundle["features"]]
    preds = bundle["booster"].predict(X_imp_df, num_iteration=bundle["best_iter"])
    calibrator = bundle["calibrator"]
    if calibrator is not None:
        preds = calibrator.transform(preds)
    return pd.Series(preds, index=fight_urls, name="parity_prob")


def main():
    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    art_dir = _find_latest_winner_artifact(cfg)
    logger.info(f"Loading winner artifacts from {art_dir}")

    meta_path_pkl = art_dir / "meta.pkl"
    meta_path_json = art_dir / "meta.json"
    if meta_path_pkl.exists():
        meta = joblib.load(meta_path_pkl)
    elif meta_path_json.exists():
        meta = json.loads(meta_path_json.read_text())
    else:
        meta = {}
    stack_threshold = float(meta.get("stack_threshold", 0.5))
    imputer: FeatureTypeImputationStrategy = joblib.load(art_dir / "imputer.pkl")
    scaler = None
    if (art_dir / "scaler.pkl").exists():
        scaler = joblib.load(art_dir / "scaler.pkl")
    features = joblib.load(art_dir / "features.pkl")["features"]

    lgb_model = lgb.Booster(model_file=str(art_dir / "lightgbm_model.txt"))
    lgb_meta = joblib.load(art_dir / "lightgbm_meta.pkl")
    lgb_best_iter = lgb_meta.get("best_iteration")
    xgb_model = xgb.Booster(model_file=str(art_dir / "xgboost_model.json"))
    xgb_meta = joblib.load(art_dir / "xgboost_meta.pkl")
    xgb_best_iter = xgb_meta.get("best_iteration")
    stacker = joblib.load(art_dir / "stacker.pkl") if (art_dir / "stacker.pkl").exists() else None

    # Load upcoming features (prefer precomputed full stack)
    loader = UFCDataLoader(cfg["paths"]["data_dir"])
    features_df = None
    features_path_cfg = cfg["paths"].get("upcoming_features")
    if features_path_cfg:
        feat_path = _resolve_path(features_path_cfg)
        if feat_path.exists():
            features_df = pd.read_parquet(feat_path)
            logger.info("Loaded upcoming feature matrix from {} (rows={}, cols={})", feat_path, len(features_df), features_df.shape[1])
        else:
            logger.warning("Configured upcoming_features path %s not found; falling back to base upcoming CSV.", feat_path)

    if features_df is not None:
        upcoming = features_df.copy()
        if "fight_url" not in upcoming.columns:
            raise ValueError("Upcoming feature matrix missing fight_url column.")
        upcoming["fight_url"] = upcoming["fight_url"].astype(str).str.strip().str.rstrip("/")
        X_up_raw = upcoming.reindex(columns=features, fill_value=np.nan)
    else:
        upcoming = loader.load_upcoming_fights(cfg["paths"]["upcoming_fights"])
        if "fight_url" in upcoming.columns:
            upcoming["fight_url"] = upcoming["fight_url"].astype(str).str.strip().str.rstrip("/")
        X_up_raw, _ = loader.prepare_features_target(
            upcoming,
            target="winner_encoded",
            exclude_odds=cfg["features"]["exclude_odds"],
        )
        X_up_raw = loader.align_to_training_features(X_up_raw.copy(), features)

    # Prepare features: align -> impute -> optional scale
    parity_bundle = _load_parity_bundle(cfg)
    fight_urls = upcoming["fight_url"] if "fight_url" in upcoming.columns else pd.Series(X_up_raw.index.astype(str))
    if parity_bundle:
        try:
            parity_series = _compute_parity_probs(loader, X_up_raw.copy(), fight_urls, parity_bundle)
            parity_probs = parity_series.reindex(fight_urls).fillna(0.5).to_numpy()
        except Exception as exc:
            logger.warning(f"Failed to compute parity probabilities for upcoming fights: {exc}")
            parity_probs = np.full(len(X_up_raw), 0.5)
    else:
        logger.warning("Parity winner artifacts not found; skipping parity probability feature for upcoming fights.")
        parity_probs = np.full(len(X_up_raw), 0.5)

    X_for_mt = X_up_raw.copy()
    X_up_aligned = loader.align_to_training_features(X_up_raw.copy(), features)
    X_up = imputer.transform(X_up_aligned)
    if scaler is not None:
        from sklearn.preprocessing import StandardScaler  # for type hints

        X_up = pd.DataFrame(scaler.transform(X_up), columns=X_up.columns, index=X_up.index)

    # Infer probabilities
    proba_lgb = lgb_model.predict(X_up, num_iteration=lgb_best_iter)
    dtest = xgb.DMatrix(X_up, feature_names=X_up.columns.tolist())
    proba_xgb = xgb_model.predict(dtest, iteration_range=(0, xgb_best_iter))
    stack_input = np.vstack([proba_lgb, proba_xgb, parity_probs]).T
    if stacker is not None:
        stack_probs = stacker.predict_proba(stack_input)[:, 1]
    else:
        stack_probs = proba_lgb
    stack_binary = (stack_probs >= stack_threshold).astype(int)

    # Assemble output
    out = upcoming.copy()
    out["pred_win_prob_f1"] = stack_probs
    out["pred_win_prob_f2"] = 1.0 - stack_probs
    out["pred_win_prob_lgb_raw"] = proba_lgb
    out["pred_win_prob_xgb_raw"] = proba_xgb
    out["stack_threshold"] = stack_threshold
    out["pred_win_binary_f1"] = stack_binary
    if {"f_1_name", "f_2_name"}.issubset(out.columns):
        out["predicted_winner"] = np.where(stack_binary == 1, out["f_1_name"], out["f_2_name"])
    out["parity_prob"] = parity_probs

    # Multi-task method and round predictions if artifacts available
    multitask_dir = ROOT / "artifacts" / "multitask"
    features_path = multitask_dir / "features.pkl"
    if multitask_dir.exists() and features_path.exists():
        mt_features = joblib.load(features_path)["features"]
        X_mt = loader.align_to_training_features(X_for_mt.copy(), mt_features)
        X_mt = X_mt.fillna(0.0)

        def _load_bundle(name: str):
            path = multitask_dir / f"{name}_lgb.pkl"
            if not path.exists():
                logger.warning(f"Missing multitask artifact: {path}")
                return None
            return joblib.load(path)

        method_bundle = _load_bundle("method")
        round_bundle = _load_bundle("round")

        if method_bundle:
            method_model = method_bundle["model"]
            method_classes = method_bundle.get("classes")
            best_iter = method_bundle.get("best_iteration") or getattr(method_model, "best_iteration", None)
            predict_kwargs = {}
            if best_iter and best_iter > 0:
                predict_kwargs["num_iteration"] = int(best_iter)
            base_method_probs = method_model.predict(X_mt, **predict_kwargs)
            method_probs = np.asarray(base_method_probs)
            calibrator = method_bundle.get("calibrator")
            calibration_input = method_bundle.get("calibration_input", "proba")
            if calibrator is not None:
                if calibration_input == "logits":
                    logits = method_model.predict(X_mt, raw_score=True, **predict_kwargs)
                    method_probs = calibrator.transform(logits)
                else:
                    method_probs = calibrator.transform(method_probs)
            method_probs = np.asarray(method_probs, dtype=float)
            if method_probs.ndim == 1:
                method_probs = method_probs.reshape(-1, 1)
            row_sums = method_probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            method_probs = method_probs / row_sums
            if method_classes is None:
                method_classes = [f"class_{i}" for i in range(method_probs.shape[1])]
            method_df = pd.DataFrame(
                method_probs,
                columns=[f"pred_method_prob_{cls}" for cls in method_classes],
                index=out.index,
            )
            out = pd.concat([out, method_df], axis=1)
            method_idx = method_probs.argmax(axis=1)
            out["pred_method"] = [method_classes[i] for i in method_idx]

        if round_bundle:
            round_model = round_bundle["model"]
            round_classes = round_bundle.get("classes")
            best_iter = round_bundle.get("best_iteration") or getattr(round_model, "best_iteration", None)
            predict_kwargs = {}
            if best_iter and best_iter > 0:
                predict_kwargs["num_iteration"] = int(best_iter)
            base_round_probs = round_model.predict(X_mt, **predict_kwargs)
            round_probs = np.asarray(base_round_probs)
            calibrator = round_bundle.get("calibrator")
            calibration_input = round_bundle.get("calibration_input", "proba")
            if calibrator is not None:
                if calibration_input == "logits":
                    logits = round_model.predict(X_mt, raw_score=True, **predict_kwargs)
                    round_probs = calibrator.transform(logits)
                else:
                    round_probs = calibrator.transform(round_probs)
            round_probs = np.asarray(round_probs, dtype=float)
            if round_probs.ndim == 1:
                round_probs = round_probs.reshape(-1, 1)
            row_sums = round_probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            round_probs = round_probs / row_sums
            if round_classes is None:
                round_classes = [f"class_{i}" for i in range(round_probs.shape[1])]
            round_df = pd.DataFrame(
                round_probs,
                columns=[f"pred_round_prob_{cls}" for cls in round_classes],
                index=out.index,
            )
            out = pd.concat([out, round_df], axis=1)
            round_idx = round_probs.argmax(axis=1)
            out["pred_finish_round"] = [round_classes[i] for i in round_idx]
    else:
        logger.warning("Multitask artifacts not found; skipping method/round predictions.")

    # Betting suggestion when odds present
    if {"f_1_odds", "f_2_odds"}.issubset(out.columns):
        # Compute Kelly on f_1 side; for f_2, mirror
        p1 = stack_probs
        o1 = out["f_1_odds"].astype(float).values
        o2 = out["f_2_odds"].astype(float).values
        f1 = kelly_fraction(p1, o1)
        # Cap and apply min edge
        cap = float(cfg["betting"]["kelly_fraction_cap"]) 
        min_edge = float(cfg["betting"]["min_edge"]) 
        edge1 = p1 * o1 - 1.0
        f1 = np.where(edge1 >= min_edge, np.minimum(f1, cap), 0.0)
        out["kelly_frac_f1"] = f1
        # Optionally compute f2 as alternative
        p2 = 1.0 - p1
        f2 = kelly_fraction(p2, o2)
        edge2 = p2 * o2 - 1.0
        f2 = np.where(edge2 >= min_edge, np.minimum(f2, cap), 0.0)
        out["kelly_frac_f2"] = f2

    # Save
    out_dir = Path(cfg["paths"]["outputs_dir"]) 
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"upcoming_predictions_{ts}.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Wrote predictions: {out_path}")


if __name__ == "__main__":
    main()
