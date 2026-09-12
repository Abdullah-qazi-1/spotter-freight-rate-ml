# Spotter Freight Rate Prediction - ML Assessment

This repository contains my solution for the Spotter Machine Learning Engineer assessment. The objective is to predict spot market truckload freight rates (`posted_rate`) based on load attributes such as origin/destination cities, distance, equipment type, weight, and date information.

---

## 1. Project Overview

The project deliverables include:
- **Video Walkthrough**: 🎥 [**Watch the 3-Minute Loom Walkthrough**](https://www.loom.com/share/6eec4905e5c8425fa58c7406c1f4b84d)
- **Validation Predictions (`validation_predictions.csv`)**: Predictions for the 12,000 unlabeled loads in `data/validation.csv` (`TE-000001` to `TE-012000`).
- **December Rate Predictions (`december_chart_inputs.csv`)**: 31-day forecast for a fixed Lexington → Fort Wayne corridor in December 2025.
- **Assessment Report (`report.pdf`)**: Technical report summarizing data exploration, validation strategy, model benchmarks, and the generated December rate chart.
- **Codebase**: Modular, reproducible training and inference code in `src/`.

---

## 2. Project Structure

```text
spotter-freight-rate-ml/
│
├── data/
│   ├── train_test.csv                       # 48,000 labeled loads (Jan 1 – Oct 31, 2025)
│   ├── validation.csv                       # 12,000 unlabeled loads (Nov 1 – Dec 31, 2025)
│   ├── validation_predictions_template.csv  # Submission template with load IDs
│   └── december_chart_inputs.csv            # 31 daily test rows (Lexington -> Fort Wayne)
│
├── src/
│   ├── __init__.py
│   ├── data.py                              # Data loading functions using pathlib
│   ├── features.py                          # Data cleaning, transformations & lane frequency encoding
│   ├── train.py                             # Training script with time-based split & CatBoost model
│   ├── predict.py                           # Batch prediction script for validation and December data
│   └── evaluate.py                          # Sanity checks and score.py runner
│
├── notebooks/
│   └── exploration.ipynb                    # Jupyter notebook with EDA and initial experiments
│
├── outputs/
│   ├── validation_predictions.csv           # Generated predictions (load_id, predicted_rate)
│   ├── december_chart_inputs.csv            # Completed December predictions
│   ├── catboost_model.cbm                   # Saved CatBoost model weights
│   ├── pipeline.joblib                      # Fitted scikit-learn preprocessor and city coordinates
│   └── feature_names.json                   # List of processed feature names
│
├── scorer_results/
│   └── candidate_december.png               # December rate chart created by score.py
│
├── validation_predictions.csv               # Top-level copy of validation predictions
├── score.py                                 # Provided assessment scorer script
├── requirements.txt                         # Required Python dependencies
├── report.pdf                               # Final assessment report (PDF)
├── README.md                                # Project documentation
└── .gitignore                               # Standard git ignore rules
```

---

## 3. Datasets

| File | Rows | Date Range | Description |
|---|:---:|:---:|---|
| `data/train_test.csv` | 48,000 | 2025-01-01 to 2025-10-31 | Labeled historical loads with `posted_rate`. |
| `data/validation.csv` | 12,000 | 2025-11-01 to 2025-12-31 | Unlabeled test set (`load_id` `TE-000001` to `TE-012000`). |
| `data/validation_predictions_template.csv` | 12,000 | — | Submission format template. |
| `data/december_chart_inputs.csv` | 31 | 2025-12-01 to 2025-12-31 | Fixed test corridor (Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb). |

---

## 4. Key EDA Findings & Data Cleaning

- **Distance is the strongest driver**: Trip distance has a ~0.91 correlation with rate. Because the relationship has slight non-linearity, `log1p(distance)` was used.
- **Target Skew**: `posted_rate` ranges from $57 to $25,533 with strong right-skew (~1.90). Training on `log1p(posted_rate)` and converting back with `expm1` stabilizes gradient updates and significantly reduces large prediction errors.
- **Negative Weights**: 292 rows contained negative weight values (e.g. -35,000 lb) due to sign-entry errors. These were fixed by taking `abs(weight)`.
- **Missing Values**: Weight was missing in 300 rows (0.62%) and `market_index` in 374 rows (0.78%). These were kept as NaN and handled with median imputation within the training pipeline.
- **Dropped Columns**: `load_id` (identifier only), `market_index`, and `quote_signal` were dropped from the final model to keep the pipeline clean and prevent noise.

---

## 5. Validation Strategy & Preventing Data Leakage

### Time-Based Split
Freight rates change over time due to seasonal demand and market conditions. Using a random train/test split would leak future information into the past.
- **Cutoff Date**: `2025-08-31` (approx. 80th percentile of labeled dates).
- **Training Set**: Dates on or before `2025-08-31` (38,477 rows, Jan–Aug 2025).
- **Internal Validation Set**: Dates after `2025-08-31` (9,523 rows, Sep–Oct 2025).

### Preventing Leakage
- Preprocessing transformers (imputers, scalers, one-hot encodings, and lane frequencies) are **fit strictly on training data**.
- `LaneFrequencyTransformer` calculates lane frequencies from training records only. Unseen lanes in validation or December data are mapped to the minimum frequency from training.
- Predictions on validation and December data use the already-fitted pipeline objects without refitting.

---

## 6. Model Comparison & Results

Models were evaluated on the internal holdout set (Sep–Oct 2025) on the original dollar scale after `expm1` inversion:

| Model | Holdout MAE ($) | Holdout RMSE ($) | Holdout R² | Notes |
|---|:---:|:---:|:---:|---|
| **CatBoost (Tuned)** | **133.0854** | **641.1042** | **0.8235** | **Best overall (lowest MAE & RMSE)** |
| CatBoost (Baseline) | 143.1829 | 644.3404 | 0.8217 | Default parameters |
| LightGBM | 145.9222 | 644.9609 | 0.8214 | Fast training, close performance |
| Linear Regression | 162.1602 | 650.9376 | 0.8181 | Simple baseline |
| Random Forest | 167.6709 | 662.6681 | 0.8114 | High variance |
| XGBoost | 167.6918 | 660.4121 | 0.8127 | Higher error on tail values |

### Final CatBoost Hyperparameters
- `depth`: 6
- `iterations`: 500
- `learning_rate`: 0.03
- `l2_leaf_reg`: 1
- `loss_function`: 'RMSE'
- `random_state`: 42

---

## 7. How to Run the Project

### Step 1: Set Up Virtual Environment (`.venv`)

**Windows (PowerShell):**
```bash
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 3: Train Model
```bash
python -m src.train
```
*Trains CatBoost on the time-based training fold and saves model artifacts to `outputs/`.*

### Step 4: Generate Predictions
```bash
python -m src.predict
```
*Creates `validation_predictions.csv` (12,000 rows) and `outputs/december_chart_inputs.csv` (31 rows).*

### Step 5: Validate Outputs & Run Official Scorer
```bash
python -m src.evaluate
```

Or run `score.py` directly:
```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

---

## 8. December Predictions Note

For the December test corridor (Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb), all physical features and the month/quarter are fixed across all 31 days. The day-to-day rate variation ($769.92 to $787.05) reflects the model's learned **day-of-week weekly cycle**, which is consistent with the sinusoidal calendar features.
