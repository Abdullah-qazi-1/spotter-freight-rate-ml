"""Feature engineering and preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
CUTOFF_DATE = pd.Timestamp("2025-08-31")
TARGET_COL = "posted_rate"
ID_COL = "load_id"
EXCLUDED_COLS = ["market_index", "quote_signal"]

NUMERIC_FEATURES = [
    "distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]
CATEGORICAL_FEATURES = ["equipment", "pickup", "delivery"]
DATE_FEATURE = "date"
LANE_FEATURE = "lane"

MODEL_COLUMNS = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "lane",
]


def correct_negative_weights(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace negative weights with their absolute value (notebook Section 18)."""
    cleaned = frame.copy()
    if "weight" in cleaned.columns:
        neg_mask = cleaned["weight"] < 0
        cleaned.loc[neg_mask, "weight"] = cleaned.loc[neg_mask, "weight"].abs()
    return cleaned


def parse_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse the date column to datetime."""
    cleaned = frame.copy()
    cleaned[DATE_FEATURE] = pd.to_datetime(cleaned[DATE_FEATURE])
    return cleaned


def add_lane(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the lane key used for frequency encoding."""
    out = frame.copy()
    out[LANE_FEATURE] = out["pickup"] + " -> " + out["delivery"]
    return out


def transform_target(y: pd.Series | np.ndarray) -> np.ndarray:
    """log1p transform used because posted_rate is right-skewed."""
    return np.log1p(y)


def inverse_target(y_log: np.ndarray) -> np.ndarray:
    """Convert model predictions back to dollars."""
    return np.expm1(y_log)


def prepare_feature_frame(frame: pd.DataFrame, *, drop_target: bool = True) -> pd.DataFrame:
    """Drop identifiers/excluded columns and add the lane feature."""
    out = frame.copy()
    drop_cols = [col for col in [ID_COL, *EXCLUDED_COLS] if col in out.columns]
    if drop_target and TARGET_COL in out.columns:
        drop_cols.append(TARGET_COL)
    out = out.drop(columns=drop_cols)
    out = add_lane(out)
    out = parse_dates(out)
    return out


def time_based_split(X: pd.DataFrame, y: pd.Series):
    """Train on dates through 2025-08-31; validate on later labeled dates."""
    train_mask = X[DATE_FEATURE] <= CUTOFF_DATE
    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    X_val = X.loc[~train_mask].copy()
    y_val = y.loc[~train_mask].copy()
    return X_train, X_val, y_train, y_val


class LaneFrequencyTransformer(BaseEstimator, TransformerMixin):
    """Frequency-encode lanes using training-fold counts only.

    Unseen lanes receive the minimum training frequency (notebook implementation).
    """

    def __init__(self, handle_unknown: str = "ignore"):
        self.handle_unknown = handle_unknown
        self.lane_frequencies = None
        self.min_freq = 0.0

    def _lane_series(self, X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            if LANE_FEATURE in X.columns:
                return X[LANE_FEATURE]
            return X.iloc[:, 0]
        if isinstance(X, pd.Series):
            return X
        return pd.Series(np.asarray(X).ravel())

    def fit(self, X, y=None):
        lanes = self._lane_series(X)
        self.lane_frequencies = lanes.value_counts(normalize=True)
        if self.handle_unknown == "ignore":
            self.min_freq = float(self.lane_frequencies.min()) if not self.lane_frequencies.empty else 0.0
        return self

    def transform(self, X):
        if self.lane_frequencies is None:
            raise RuntimeError("LaneFrequencyTransformer has not been fitted yet.")
        lanes = self._lane_series(X)
        transformed = lanes.map(self.lane_frequencies).fillna(self.min_freq)
        return transformed.to_frame()


def create_cyclical_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """sin/cos encodings for month, quarter, and day of week."""
    X = X_df.copy()
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[DATE_FEATURE])
    if DATE_FEATURE not in X.columns:
        X = X.copy()
        X.columns = [DATE_FEATURE]

    if not pd.api.types.is_datetime64_any_dtype(X[DATE_FEATURE]):
        X[DATE_FEATURE] = pd.to_datetime(X[DATE_FEATURE])

    month = X[DATE_FEATURE].dt.month
    quarter = X[DATE_FEATURE].dt.quarter
    day_of_week = X[DATE_FEATURE].dt.dayofweek

    return pd.DataFrame(
        {
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
            "quarter_sin": np.sin(2 * np.pi * quarter / 4.0),
            "quarter_cos": np.cos(2 * np.pi * quarter / 4.0),
            "day_of_week_sin": np.sin(2 * np.pi * day_of_week / 7.0),
            "day_of_week_cos": np.cos(2 * np.pi * day_of_week / 7.0),
        },
        index=X.index,
    )


def build_preprocessor() -> ColumnTransformer:
    """Return the unfitted ColumnTransformer from the notebook."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    distance_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    cyclical_date_transformer = Pipeline(
        steps=[
            ("extract_cyclical", FunctionTransformer(create_cyclical_features, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("distance_pipe", distance_transformer, ["distance"]),
            ("num", numeric_transformer, [f for f in NUMERIC_FEATURES if f != "distance"]),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("lane_freq", LaneFrequencyTransformer(), [LANE_FEATURE]),
            ("cyclical_date", cyclical_date_transformer, [DATE_FEATURE]),
        ],
        remainder="drop",
    )
    return preprocessor


def processed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Column names after preprocessing, matching the notebook order."""
    num_feat_names = ["distance_log1p"] + [f for f in NUMERIC_FEATURES if f != "distance"]
    cat_feature_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(
        CATEGORICAL_FEATURES
    )
    lane_freq_name = ["lane_frequency"]
    cyclical_date_names = [
        "month_sin",
        "month_cos",
        "quarter_sin",
        "quarter_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ]
    return num_feat_names + list(cat_feature_names) + lane_freq_name + cyclical_date_names


def transform_features(preprocessor: ColumnTransformer, X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Apply a fitted preprocessor and return a named dense DataFrame."""
    processed = preprocessor.transform(X)
    if hasattr(processed, "toarray"):
        processed = processed.toarray()
    return pd.DataFrame(processed, columns=feature_names, index=X.index)


def city_coordinate_maps(labeled_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """City -> coordinate lookup used for December rows (from cleaned labeled data)."""
    pickup_coords_map = labeled_frame[["pickup", "pickup_lat", "pickup_lon"]].drop_duplicates()
    delivery_coords_map = labeled_frame[["delivery", "delivery_lat", "delivery_lon"]].drop_duplicates()
    return pickup_coords_map, delivery_coords_map


def add_coordinates_from_maps(
    frame: pd.DataFrame,
    pickup_coords_map: pd.DataFrame,
    delivery_coords_map: pd.DataFrame,
) -> pd.DataFrame:
    """Merge training city coordinates onto December (or other) rows."""
    out = frame.merge(pickup_coords_map, on="pickup", how="left")
    out = out.merge(delivery_coords_map, on="delivery", how="left")
    return out
