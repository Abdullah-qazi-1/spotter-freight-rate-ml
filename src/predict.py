"""Generate official validation and December prediction files."""

from __future__ import annotations

import joblib
import pandas as pd
from catboost import CatBoostRegressor

from src.data import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT, load_december, load_validation
from src.features import (
    MODEL_COLUMNS,
    add_coordinates_from_maps,
    add_lane,
    inverse_target,
    parse_dates,
    prepare_feature_frame,
    transform_features,
)

DECEMBER_OUTPUT_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]


def _load_artifacts():
    artifact_path = MODELS_DIR / "pipeline.joblib"
    model_path = MODELS_DIR / "catboost_model.cbm"
    if not artifact_path.is_file() or not model_path.is_file():
        raise SystemExit("Missing trained artifacts. Run `python -m src.train` first.")
    artifacts = joblib.load(artifact_path)
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return artifacts, model


def predict_validation(artifacts, model) -> pd.DataFrame:
    df_val_raw = load_validation()
    X = prepare_feature_frame(df_val_raw, drop_target=True)
    processed = transform_features(artifacts["preprocessor"], X, artifacts["feature_names"])
    predicted_rate = inverse_target(model.predict(processed))
    return pd.DataFrame(
        {
            "load_id": df_val_raw["load_id"],
            "predicted_rate": predicted_rate,
        }
    )


def predict_december(artifacts, model) -> pd.DataFrame:
    df_december_raw = load_december()
    df_december = add_lane(df_december_raw)
    df_december = parse_dates(df_december)

    with_coords = add_coordinates_from_maps(
        df_december,
        artifacts["pickup_coords_map"],
        artifacts["delivery_coords_map"],
    )
    if len(with_coords) != len(df_december):
        raise SystemExit(
            "December coordinate merge changed the row count. "
            "City coordinate maps must be unique per city."
        )

    missing_coords = with_coords[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].isna().any().any()
    if missing_coords:
        print("WARNING: some December cities were not found in training coordinates.")

    X = with_coords[MODEL_COLUMNS]
    processed = transform_features(artifacts["preprocessor"], X, artifacts["feature_names"])
    predicted_rate = inverse_target(model.predict(processed))

    output = df_december_raw.copy()
    output["predicted_rate"] = predicted_rate
    return output[DECEMBER_OUTPUT_COLUMNS]


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts, model = _load_artifacts()

    val_preds = predict_validation(artifacts, model)
    val_path = OUTPUTS_DIR / "validation_predictions.csv"
    val_preds.to_csv(val_path, index=False)
    val_root_path = PROJECT_ROOT / "validation_predictions.csv"
    val_preds.to_csv(val_root_path, index=False)
    print(f"Wrote {len(val_preds):,} validation rows to {val_path} and {val_root_path}")

    december_preds = predict_december(artifacts, model)
    december_path = OUTPUTS_DIR / "december_chart_inputs.csv"
    december_preds.to_csv(december_path, index=False)
    december_data_path = DATA_DIR / "december_chart_inputs.csv"
    december_preds.to_csv(december_data_path, index=False)
    print(f"Wrote {len(december_preds):,} December rows to {december_path} and {december_data_path}")


if __name__ == "__main__":
    main()
