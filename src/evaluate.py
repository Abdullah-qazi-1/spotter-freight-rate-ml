"""Check prediction-file contracts and run the official scorer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import OUTPUTS_DIR, PROJECT_ROOT, SCORER_RESULTS_DIR, load_validation

EXPECTED_VAL_ROWS = 12_000
EXPECTED_IDS = {f"TE-{index:06d}" for index in range(1, EXPECTED_VAL_ROWS + 1)}
DECEMBER_DATES = pd.date_range("2025-12-01", "2025-12-31", freq="D")
FIXED_PICKUP = "Lexington"
FIXED_DELIVERY = "Fort Wayne"
FIXED_DISTANCE = 360.0
FIXED_EQUIPMENT = "Dry Van"
FIXED_WEIGHT = 32_000.0


def _fail(message: str) -> None:
    raise SystemExit(f"EVALUATION FAILED: {message}")


def check_validation_predictions(path: Path) -> None:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["load_id", "predicted_rate"]:
        _fail("validation predictions must have columns load_id,predicted_rate")
    if len(frame) != EXPECTED_VAL_ROWS:
        _fail(f"validation predictions must have {EXPECTED_VAL_ROWS:,} rows")
    if frame["load_id"].isna().any() or frame["load_id"].duplicated().any():
        _fail("load_id is missing or duplicated")

    submitted = set(frame["load_id"].astype(str))
    if submitted != EXPECTED_IDS:
        _fail(
            "load_id values do not match the validation set "
            f"(missing={len(EXPECTED_IDS - submitted)}, extra={len(submitted - EXPECTED_IDS)})"
        )

    official_ids = set(load_validation()["load_id"].astype(str))
    if submitted != official_ids:
        _fail("load_id values do not match validation.csv")

    rates = pd.to_numeric(frame["predicted_rate"], errors="coerce")
    if rates.isna().any() or not np.isfinite(rates).all():
        _fail("predicted_rate contains missing or non-finite values")
    if (rates <= 0).any():
        _fail("predicted_rate contains non-positive values")
    print(f"Validation predictions OK ({len(frame):,} rows).")
    print(f"  predicted_rate range: {rates.min():.2f} to {rates.max():.2f}")


def check_december_predictions(path: Path) -> None:
    frame = pd.read_csv(path)
    columns = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    if list(frame.columns) != columns:
        _fail("December file must keep the original seven columns in order")
    if len(frame) != 31:
        _fail("December file must contain exactly 31 rows")

    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        _fail("December dates are missing or duplicated")
    if set(dates) != set(DECEMBER_DATES):
        _fail("December file must contain each day from 2025-12-01 to 2025-12-31 once")
    if not frame["pickup"].eq(FIXED_PICKUP).all():
        _fail(f"pickup must stay {FIXED_PICKUP}")
    if not frame["delivery"].eq(FIXED_DELIVERY).all():
        _fail(f"delivery must stay {FIXED_DELIVERY}")
    if not np.isclose(pd.to_numeric(frame["distance"]), FIXED_DISTANCE).all():
        _fail(f"distance must stay {FIXED_DISTANCE:g}")
    if not frame["equipment"].eq(FIXED_EQUIPMENT).all():
        _fail(f"equipment must stay {FIXED_EQUIPMENT}")
    if not np.isclose(pd.to_numeric(frame["weight"]), FIXED_WEIGHT).all():
        _fail(f"weight must stay {FIXED_WEIGHT:g}")

    rates = pd.to_numeric(frame["predicted_rate"], errors="coerce")
    if rates.isna().any() or not np.isfinite(rates).all():
        _fail("December predicted_rate contains missing or non-finite values")
    if (rates <= 0).any():
        _fail("December predicted_rate contains non-positive values")
    print("December predictions OK (31 rows).")
    print(f"  predicted_rate range: {rates.min():.2f} to {rates.max():.2f}")


def run_official_scorer(val_path: Path, december_path: Path) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "score.py"),
        "--predictions",
        str(val_path),
        "--december-predictions",
        str(december_path),
        "--output-dir",
        str(SCORER_RESULTS_DIR),
    ]
    print("\nRunning official scorer:")
    print(" ", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        _fail(f"score.py exited with code {result.returncode}")
    chart = SCORER_RESULTS_DIR / "candidate_december.png"
    if not chart.is_file():
        _fail("score.py did not create scorer_results/candidate_december.png")
    print(f"Official scorer OK. Chart: {chart}")


def main() -> None:
    val_path = OUTPUTS_DIR / "validation_predictions.csv"
    val_root_path = PROJECT_ROOT / "validation_predictions.csv"
    december_path = OUTPUTS_DIR / "december_chart_inputs.csv"
    if not val_path.is_file() or not december_path.is_file():
        _fail("Prediction files are missing. Run `python -m src.predict` first.")

    check_validation_predictions(val_path)
    if val_root_path.is_file():
        check_validation_predictions(val_root_path)
    check_december_predictions(december_path)
    run_official_scorer(val_path, december_path)


if __name__ == "__main__":
    main()
