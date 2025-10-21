"""
CI-style smoke runner that enforces the critical safeguards:
  1. Great Expectations validations (must pass).
  2. Enhanced winner training with accuracy guards.
  3. Parity check on legacy golden features with minimum accuracy targets.

Exit code is non-zero if any step fails or if parity metrics fall below thresholds.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import subprocess

from loguru import logger

ROOT = Path(__file__).parents[1]


@dataclass
class Step:
    name: str
    command: Sequence[str]


def run_step(step: Step) -> None:
    logger.info(f"Running step: {step.name}")
    proc = subprocess.run(step.command, cwd=ROOT, text=True)
    if proc.returncode != 0:
        logger.error(f"Step '{step.name}' failed with code {proc.returncode}")
        raise SystemExit(proc.returncode)


def assert_parity_metrics(summary_path: Path, min_val_acc: float, min_test_acc: float) -> None:
    if not summary_path.exists():
        logger.error(f"Parity summary not found: {summary_path}")
        raise SystemExit(10)
    with summary_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        val_acc = None
        test_acc = None
        for row in reader:
            if row["model"] == "lightgbm_tuned" and row["split"] == "val" and row["calibrated"] == "no":
                val_acc = float(row["accuracy"])
            if row["model"] == "lightgbm_tuned" and row["split"] == "test" and row["calibrated"] == "no":
                test_acc = float(row["accuracy"])
        logger.info(f"Parity metrics: val_acc={val_acc}, test_acc={test_acc}")
        if val_acc is None or val_acc < min_val_acc:
            logger.error(f"Parity val accuracy {val_acc} below minimum {min_val_acc}")
            raise SystemExit(11)
        if test_acc is None or test_acc < min_test_acc:
            logger.error(f"Parity test accuracy {test_acc} below minimum {min_test_acc}")
            raise SystemExit(12)


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    steps = [
        Step("Great Expectations", [sys.executable, str(ROOT / "scripts" / "run_ge_validations.py")]),
        Step(
            "Winner Enhanced Guarded",
            [
                sys.executable,
                str(ROOT / "scripts" / "train_winner_enhanced.py"),
                "--min-val-acc",
                "0.71",
                "--min-test-acc",
                "0.62",
            ],
        ),
        Step("Winner Parity", [sys.executable, str(ROOT / "scripts" / "train_winner_parity.py")]),
    ]

    for step in steps:
        run_step(step)

    assert_parity_metrics(ROOT / "outputs" / "parity_winner_summary.csv", min_val_acc=0.72, min_test_acc=0.60)
    logger.info("CI smoke run completed successfully.")


if __name__ == "__main__":
    main()
