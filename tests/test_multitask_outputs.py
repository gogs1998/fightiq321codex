import unittest
from pathlib import Path

import importlib.util

import joblib
import numpy as np

from src.utils.config import load_config


class TestMultitaskOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parents[1]
        cls.config = load_config(cls.root / "config" / "config.yaml")
        cls.artifacts_dir = cls.root / "artifacts" / "multitask"
        if not cls.artifacts_dir.exists():
            raise unittest.SkipTest("Multitask artifacts missing; run train_multitask.py first.")

        features_path = cls.artifacts_dir / "features.pkl"
        if not features_path.exists():
            raise unittest.SkipTest("Multitask feature manifest missing.")
        cls.feature_list = joblib.load(features_path)["features"]

        module_path = cls.root / "scripts" / "train_multitask.py"
        if not module_path.exists():
            raise unittest.SkipTest("train_multitask.py not found for feature loading.")
        spec = importlib.util.spec_from_file_location("train_multitask_module", module_path)
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise unittest.SkipTest("Unable to load train_multitask module.")
        spec.loader.exec_module(module)
        meta, X_full, *_ = module._load_features_and_labels(cls.config)
        if X_full.empty:
            raise unittest.SkipTest("Training features empty; rerun train_multitask.py.")
        cls.sample = X_full[cls.feature_list].head(64).fillna(0.0)

    def _load_bundle(self, name: str):
        path = self.artifacts_dir / f"{name}_lgb.pkl"
        if not path.exists():
            self.skipTest(f"{name} bundle missing at {path}")
        return joblib.load(path)

    def _assert_probs_sum(self, probs: np.ndarray):
        row_sums = probs.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities must sum to 1 per row.")

    def test_method_probabilities_calibrated(self):
        bundle = self._load_bundle("method")
        model = bundle["model"]
        best_iter = bundle.get("best_iteration") or getattr(model, "best_iteration", None)
        predict_kwargs = {"num_iteration": int(best_iter)} if best_iter and best_iter > 0 else {}
        base_probs = model.predict(self.sample, **predict_kwargs)
        calibrator = bundle.get("calibrator")
        calibration_input = bundle.get("calibration_input", "proba")
        if calibrator is not None and calibration_input == "logits":
            logits = model.predict(self.sample, raw_score=True, **predict_kwargs)
            probs = calibrator.transform(logits)
        elif calibrator is not None:
            probs = calibrator.transform(base_probs)
        else:
            probs = base_probs
        probs = np.asarray(probs, dtype=float)
        self._assert_probs_sum(probs)

    def test_round_probabilities_calibrated(self):
        bundle = self._load_bundle("round")
        model = bundle["model"]
        best_iter = bundle.get("best_iteration") or getattr(model, "best_iteration", None)
        predict_kwargs = {"num_iteration": int(best_iter)} if best_iter and best_iter > 0 else {}
        base_probs = model.predict(self.sample, **predict_kwargs)
        calibrator = bundle.get("calibrator")
        calibration_input = bundle.get("calibration_input", "proba")
        if calibrator is not None and calibration_input == "logits":
            logits = model.predict(self.sample, raw_score=True, **predict_kwargs)
            probs = calibrator.transform(logits)
        elif calibrator is not None:
            probs = calibrator.transform(base_probs)
        else:
            probs = base_probs
        probs = np.asarray(probs, dtype=float)
        self._assert_probs_sum(probs)
