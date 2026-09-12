"""Load assessment CSV files from the project data folder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCORER_RESULTS_DIR = PROJECT_ROOT / "scorer_results"
MODELS_DIR = OUTPUTS_DIR


def load_train_test(data_dir: Path | None = None) -> pd.DataFrame:
    """Load labeled training data (`posted_rate` is present)."""
    path = (data_dir or DATA_DIR) / "train_test.csv"
    return pd.read_csv(path)


def load_validation(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the official unlabeled validation set."""
    path = (data_dir or DATA_DIR) / "validation.csv"
    return pd.read_csv(path)


def load_december(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the 31-row Lexington -> Fort Wayne December chart inputs."""
    path = (data_dir or DATA_DIR) / "december_chart_inputs.csv"
    return pd.read_csv(path)


def load_validation_template(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the official `load_id,predicted_rate` template."""
    path = (data_dir or DATA_DIR) / "validation_predictions_template.csv"
    return pd.read_csv(path)
