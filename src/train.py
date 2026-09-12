"""Train the final CatBoost model from the notebook."""

from __future__ import annotations

import json

import joblib
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import MODELS_DIR, OUTPUTS_DIR, load_train_test
from src.features import (
    RANDOM_STATE,
    TARGET_COL,
    build_preprocessor,
    city_coordinate_maps,
    correct_negative_weights,
    inverse_target,
    prepare_feature_frame,
    processed_feature_names,
    time_based_split,
    transform_features,
    transform_target,
)

# Exact CatBoost GridSearch winners from the notebook.
CATBOOST_PARAMS = {
    "depth": 6,
    "iterations": 500,
    "l2_leaf_reg": 1,
    "learning_rate": 0.03,
    "loss_function": "RMSE",
    "random_state": RANDOM_STATE,
    "verbose": 0,
    "allow_writing_files": False,
}


def _metrics(y_true_dollars: np.ndarray, y_pred_dollars: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true_dollars, y_pred_dollars)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_dollars, y_pred_dollars))),
        "r2": float(r2_score(y_true_dollars, y_pred_dollars)),
    }


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_train_test()
    cleaned = correct_negative_weights(raw)

    y = transform_target(cleaned[TARGET_COL])
    X = prepare_feature_frame(cleaned, drop_target=True)
    y = y.loc[X.index]

    X_train, X_val, y_train, y_val = time_based_split(X, y)
    print(f"Training rows (date <= 2025-08-31): {len(X_train):,}")
    print(f"Internal validation rows (date > 2025-08-31): {len(X_val):,}")

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    feature_names = processed_feature_names(preprocessor)

    X_train_processed = transform_features(preprocessor, X_train, feature_names)
    X_val_processed = transform_features(preprocessor, X_val, feature_names)

    print(f"Processed feature count: {len(feature_names)}")
    print("Fitting preprocessor on training rows only (lane frequency, medians, scalers, one-hot).")

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    print("Training CatBoost with notebook hyperparameters:")
    print(CATBOOST_PARAMS)
    model.fit(X_train_processed, y_train)

    y_val_pred = inverse_target(model.predict(X_val_processed))
    y_val_true = inverse_target(y_val)
    metrics = _metrics(y_val_true, y_val_pred)

    print("\nInternal time-based validation (original dollar scale after expm1):")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R^2:  {metrics['r2']:.4f}")
    print("Notebook reference for the same CatBoost configuration: MAE 133.0854, RMSE 641.1042, R^2 0.8235")

    pickup_map, delivery_map = city_coordinate_maps(cleaned)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "feature_names": feature_names,
            "pickup_coords_map": pickup_map,
            "delivery_coords_map": delivery_map,
            "catboost_params": CATBOOST_PARAMS,
            "internal_val_metrics": metrics,
        },
        MODELS_DIR / "pipeline.joblib",
    )
    model.save_model(str(MODELS_DIR / "catboost_model.cbm"))
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    print(f"\nSaved artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()
