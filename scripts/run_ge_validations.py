"""
Run Great Expectations validations for raw/silver data if GE is available.
If Great Expectations is not installed, prints guidance and exits 0.

Usage:
  python fightiq_codex/scripts/run_ge_validations.py
"""

import sys
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.validation.ge_suites import SUITES, DatasetSuite


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    for base in [ROOT, ROOT.parent]:
        candidate = base / path
        if candidate.exists():
            return candidate
    return ROOT / path


def _dataset_path(suite: DatasetSuite, cfg) -> Path | None:
    if suite.category == "raw":
        if cfg["ingestion"]["sink"].lower() != "parquet":
            return None
        raw_dir = _resolve_path(cfg["paths"]["raw_dir"])
        return raw_dir / suite.filename
    if suite.category in ("silver", "gold"):
        base_dir = ROOT / "data"
        return base_dir / suite.filename
    return None


def main():
    try:
        import great_expectations as ge  # type: ignore
        from great_expectations.dataset import PandasDataset  # type: ignore
    except Exception:
        print("Great Expectations not installed. Skipping GE validations. Optional: pip install great-expectations.")
        return

    import pandas as pd

    cfg = load_config(ROOT / "config/config.yaml")
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("logging", {}).get("level", "INFO"))

    failures = 0
    summary = []
    for suite in SUITES:
        path = _dataset_path(suite, cfg)
        if not path:
            logger.warning(f"Skipping {suite.name}: unsupported sink or base path.")
            continue
        if not path.exists():
            logger.warning(f"Skipping {suite.name}: {path} not found")
            continue
        pdf = pd.read_parquet(path)
        for exp in suite.expectations:
            if exp.expectation_type == "expect_column_values_to_be_between":
                col = exp.kwargs.get("column")
                if col and col in pdf.columns:
                    pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
            if exp.expectation_type == "expect_column_values_to_be_dateutil_parseable":
                col = exp.kwargs.get("column")
                if col and col in pdf.columns:
                    pdf[col] = pdf[col].astype(str)
        df = PandasDataset(pdf)
        logger.info(f"Validating {suite.name} ({len(df)} rows)")
        dataset_report = {
            "dataset": suite.name,
            "path": str(path),
            "rows": int(len(df)),
            "expectations": len(suite.expectations),
            "failed_expectations": [],
        }
        for exp in suite.expectations:
            method = getattr(df, exp.expectation_type, None)
            if method is None:
                logger.error(f"{suite.name}: expectation {exp.expectation_type} not available")
                failures += 1
                dataset_report["failed_expectations"].append(
                    {"expectation_type": exp.expectation_type, "error": "not_available", "kwargs": exp.kwargs}
                )
                continue
            res = method(**exp.kwargs)
            if not res.success:
                logger.error(f"{suite.name}: expectation {exp.expectation_type} failed with result={res}")
                failures += 1
                try:
                    dataset_report["failed_expectations"].append(res.to_json_dict())
                except Exception:
                    dataset_report["failed_expectations"].append(
                        {"expectation_type": exp.expectation_type, "kwargs": exp.kwargs, "result": str(res)}
                    )
        summary.append(dataset_report)

    out = ROOT / 'outputs' / 'ge_summary.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    out.write_text(_json.dumps({'failures': failures, 'datasets': summary}, indent=2))
    print(f"Wrote GE summary: {out}")

    if failures:
        logger.error(f"GE validations completed with {failures} failures")
        sys.exit(1)
    else:
        logger.info("GE validations passed")


if __name__ == '__main__':
    main()
