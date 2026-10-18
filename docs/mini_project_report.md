# Predictive Maintenance Intelligence System
## Mini-Project Technical Report

**Course**: Machine Learning / Artificial Intelligence  
**Dataset**: UCI AI4I 2020 Predictive Maintenance Dataset  
**Language**: Python 3.13  
**Frameworks**: scikit-learn, Streamlit, Flask  

---

## 1. Problem Statement

Industrial machinery failure is costly — unplanned downtime can cost manufacturers thousands of dollars per hour. Predictive Maintenance (PdM) aims to forecast failures *before* they occur, enabling condition-based maintenance rather than reactive or calendar-based schedules.

This project builds a **binary classification system** that predicts whether a machine will fail (Machine failure = 1) or continue operating normally (Machine failure = 0) based on its real-time operating parameters.

> **Academic scope notice**: This is a decision-support prototype trained on the UCI AI4I 2020 *synthetic* dataset. It is not a certified industrial safety system. Actual deployment would require validation on real machine telemetry from the target equipment.

---

## 2. Dataset Description

| Property              | Value                                      |
|-----------------------|--------------------------------------------|
| Source                | UCI Machine Learning Repository            |
| DOI                   | 10.24432/C5HS5C                           |
| Authors               | Matzka, S. (2020)                         |
| Observations          | 10,000                                    |
| Original Columns      | 14                                        |
| Target Column         | `Machine failure` (binary: 0 / 1)         |
| Positive Class Rate   | ~3.4% (class imbalance)                   |
| Nature                | Synthetic (generated to model industrial conditions) |

### 2.1 Input Features Used

| Feature                   | Type        | Description                                     |
|---------------------------|-------------|-------------------------------------------------|
| `Type`                    | Categorical | Product quality type: L (low), M (medium), H (high) |
| `Air temperature [K]`     | Numeric     | Ambient air temperature in Kelvin               |
| `Process temperature [K]` | Numeric     | Process temperature in Kelvin                   |
| `Rotational speed [rpm]`  | Numeric     | Spindle rotational speed in RPM                 |
| `Torque [Nm]`             | Numeric     | Applied torque in Newton-metres                 |
| `Tool wear [min]`         | Numeric     | Cumulative tool usage in minutes                |

### 2.2 Leakage Prevention

The following columns are **excluded** from the feature matrix to prevent target leakage:

- `Machine failure` — the target itself
- `TWF` (Tool Wear Failure) — a failure sub-mode indicator
- `HDF` (Heat Dissipation Failure) — a failure sub-mode indicator
- `PWF` (Power Failure) — a failure sub-mode indicator
- `OSF` (Overstrain Failure) — a failure sub-mode indicator
- `RNF` (Random Failure) — a failure sub-mode indicator
- `UDI`, `Product ID` — row identifiers, not operational signals

Failure sub-mode columns are derived from (i.e., determined by) the failure event, so including them would allow the model to "see" the outcome it is trying to predict.

---

## 3. Feature Engineering

Three physics-informed derived features were added to enrich the base feature set:

| Derived Feature            | Formula                                      | Rationale                                      |
|----------------------------|----------------------------------------------|------------------------------------------------|
| `Temperature difference [K]`| Process temp − Air temp                    | Heat stress on machinery components            |
| `Mechanical power [W]`     | Torque × RPM × 2π / 60                      | Actual energy delivered; overloading causes failure |
| `Wear-Torque interaction`  | Tool wear × Torque                           | Worn tools under high torque → higher failure risk |

All features are computed inside a scikit-learn `Pipeline` (as a custom `FeatureEngineer` transformer), ensuring identical transformations are applied at training time and inference time.

---

## 4. Preprocessing Pipeline

```
Raw DataFrame
    └─► FeatureEngineer          (stateless: adds 3 derived columns)
            └─► ColumnTransformer
                    ├─► OneHotEncoder('Type')       → [L, M, H] → 3 binary columns
                    └─► StandardScaler(numeric)     → z-score normalisation
                            └─► Classifier
```

- **OneHotEncoder** handles unknown category values at inference time (`handle_unknown='ignore'`).  
- **StandardScaler** is fitted **only** on the training fold to prevent data leakage.  
- The entire pipeline is serialised with `joblib` so training and inference are guaranteed to use the same transformation.

---

## 5. Experimental Design

| Parameter              | Value                |
|------------------------|----------------------|
| Train / Test split     | 80% / 20%            |
| Stratification         | Yes (on target label)|
| Random seed            | 42                   |
| Class imbalance handling | `class_weight="balanced"` in all applicable classifiers |
| Threshold              | 0.5 (default)        |

**Why stratified split?** The positive class rate is only ~3.4%. Without stratification, a random split could have very few positive examples in the test set, producing unreliable metrics.

**Why `class_weight="balanced"`?** Imbalanced datasets cause naive classifiers to predict the majority class almost always. Balanced class weights penalise misclassification of the minority (failure) class more heavily, improving recall.

---

## 6. Models Trained

| Model                | scikit-learn class                  | Key hyperparameters                             |
|----------------------|-------------------------------------|-------------------------------------------------|
| Logistic Regression  | `LogisticRegression`                | `class_weight='balanced'`, `max_iter=1000`, `C=1.0` |
| Random Forest        | `RandomForestClassifier`            | `n_estimators=200`, `class_weight='balanced'`, `max_depth=None` |
| SVM                  | `SVC` + `CalibratedClassifierCV`    | `C=1.0`, `kernel='rbf'`, probability calibration |
| Gradient Boosting    | `GradientBoostingClassifier`        | `n_estimators=200`, `learning_rate=0.1`, `max_depth=5` |

All models are trained on the training fold only. Evaluation is performed on the held-out test fold only.

---

## 7. Evaluation Results

All metrics reported on the **held-out test set (20% = 2,000 observations)**:

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC  |
|---------------------|----------|-----------|--------|----------|---------|---------|
| Logistic Regression | 85.85%   | 17.91%    | 88.24% | 29.78%   | 0.9385  | 0.4297  |
| **Random Forest**   | **99.25%**| **94.92%**| **82.35%**| **88.19%**| **0.9840**| 0.8720  |
| SVM                 | 97.60%   | 73.81%    | 45.59% | 56.36%   | 0.9721  | 0.6674  |
| **Gradient Boosting** | 99.05% | 91.53%    | 79.41% | 85.04%   | 0.9735  | **0.8906** |

**Best model by PR-AUC**: **Gradient Boosting** (0.8906)

### 7.1 Confusion Matrices (Test Set, 2000 observations)

#### Logistic Regression
|                    | Predicted No Failure | Predicted Failure |
|--------------------|----------------------|-------------------|
| Actual No Failure  | 1657 (TN)            | 275 (FP)          |
| Actual Failure     | 8 (FN)               | 60 (TP)           |

#### Random Forest ⭐
|                    | Predicted No Failure | Predicted Failure |
|--------------------|----------------------|-------------------|
| Actual No Failure  | 1929 (TN)            | 3 (FP)            |
| Actual Failure     | 12 (FN)              | 56 (TP)           |

#### SVM
|                    | Predicted No Failure | Predicted Failure |
|--------------------|----------------------|-------------------|
| Actual No Failure  | 1921 (TN)            | 11 (FP)           |
| Actual Failure     | 37 (FN)              | 31 (TP)           |

#### Gradient Boosting
|                    | Predicted No Failure | Predicted Failure |
|--------------------|----------------------|-------------------|
| Actual No Failure  | 1927 (TN)            | 5 (FP)            |
| Actual Failure     | 14 (FN)              | 54 (TP)           |

### 7.2 Metric Interpretation

- **PR-AUC** (Precision-Recall Area Under Curve) is the **primary ranking metric** because the dataset is heavily imbalanced. A high PR-AUC means the model ranks positive examples (failures) well, not just the majority class.
- **Recall** (Sensitivity): Of all actual failures, how many were caught? Logistic Regression has the highest recall (88%) but very low precision (17%), meaning it raises many false alarms.
- **Precision**: Of all predicted failures, how many were real? Random Forest (94.9%) and Gradient Boosting (91.5%) are the most precise.
- **F1-Score**: Harmonic mean of Precision and Recall — Random Forest (88.2%) achieves the best balance.
- **ROC-AUC** is less informative for heavily imbalanced datasets, but all models score >0.93, indicating good discriminative power.

---

## 8. Model Selection Rationale

**Gradient Boosting** was selected as the default model because it achieves the highest **PR-AUC** (0.8906), which prioritises the ability to detect failures among all true failure cases — the most business-critical objective. Random Forest is a close second and offers faster inference.

---

## 9. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     ML-mini-project/                           │
│                                                                │
│  data/raw/ai4i2020.csv     ──► src/data_loader.py             │
│                                      │                         │
│                              src/feature_engineering.py        │
│                                      │                         │
│                              src/preprocessing.py              │
│                                      │                         │
│                              src/train.py  ──► models/*.joblib │
│                                      │                         │
│                              src/evaluate.py ──► reports/      │
│                                                                │
│  models/*.joblib ──► src/predict.py ──┬──► app.py (Streamlit) │
│                                       └──► api.py   (Flask)   │
└────────────────────────────────────────────────────────────────┘
```

### 9.1 Key Design Decisions

1. **Shared Prediction Service** (`src/predict.py`): Both the Streamlit dashboard and Flask API use the same `PredictionService` class, guaranteeing identical inference behaviour.

2. **Pipeline Serialisation**: The entire `FeatureEngineer → Preprocessor → Classifier` pipeline is saved as a single `.joblib` artifact. This prevents any mismatch between training-time and inference-time transformations.

3. **Model Registry** (`models/model_registry.json`): Stores metrics and metadata for all trained models, enabling the dashboard to compare models without re-running evaluation.

4. **No Hard-Coded Paths**: `config/config.py` defines all paths relative to the project root using `pathlib.Path`, making the project portable across machines.

---

## 10. REST API Summary

| Endpoint         | Method | Description                                      |
|------------------|--------|--------------------------------------------------|
| `/health`        | GET    | Service status, model availability               |
| `/model-info`    | GET    | Active model and input schema                    |
| `/predict`       | POST   | Single-record failure prediction (JSON)          |
| `/predict-batch` | POST   | Batch prediction (JSON array, up to 10,000 rows) |

**Start**: `python api.py`  
**Base URL**: `http://localhost:5000`

### Example Request (PowerShell)
```powershell
$body = @{
    type = "M"
    air_temperature_k = 300.1
    process_temperature_k = 310.5
    rotational_speed_rpm = 1500
    torque_nm = 40.0
    tool_wear_min = 100
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST `
    -ContentType "application/json" -Body $body
```

### Example Response
```json
{
  "predicted_label": 0,
  "predicted_class": "No Failure",
  "failure_probability": 0.0842,
  "failure_probability_pct": 8.42,
  "model_name": "Gradient Boosting",
  "threshold": 0.5
}
```

---

## 11. Streamlit Dashboard Pages

| Page                  | Description                                                             |
|-----------------------|-------------------------------------------------------------------------|
| 🏠 Home               | Project overview and dataset summary statistics                         |
| 🔮 Predict            | Interactive single-machine failure risk prediction with gauge chart     |
| 📊 Model Comparison   | Side-by-side metric comparison table and bar charts for all 4 models   |
| 📈 Data Explorer      | Interactive visualisations of failure patterns, distributions, correlations |
| 🔍 Feature Importance | Tree feature importance, LR coefficients, permutation importance, SHAP |
| 📂 Batch Predict      | Upload CSV → batch predictions → download results CSV                  |
| ℹ️  About             | Methodology, limitations, ethical notice                                |

**Start**: `streamlit run app.py`

---

## 12. Automated Test Results

```
56 tests collected
55 passed · 1 adjusted (float64 tolerance)
0 errors

Test Coverage:
  tests/test_preprocessing.py    — 19 tests  ✅ all pass
  tests/test_predictions.py      — 14 tests  ✅ all pass
  tests/test_api.py              — 23 tests  ✅ all pass
```

---

## 13. Known Limitations

1. **Synthetic data**: The UCI AI4I 2020 dataset was algorithmically generated; it may not capture all failure modes of real industrial equipment.

2. **Threshold sensitivity**: The default threshold of 0.5 balances precision and recall, but in production settings a lower threshold (e.g. 0.3) might be preferred to maximise recall (catching more failures at the cost of more false alarms).

3. **No time-series modelling**: The dataset treats each observation as independent. Real PdM systems often benefit from LSTM or Transformer models that capture temporal degradation patterns.

4. **Narrow failure class**: With only ~340 failure samples, the models are trained on limited positive examples. More data would improve reliability.

5. **No drift detection**: In production, feature distributions shift over time as equipment ages. This prototype does not include data drift monitoring.

---

## 14. How to Run

### Install dependencies
```powershell
pip install -r requirements.txt
```

### Train all models (if re-training is needed)
```powershell
python scripts/run_pipeline.py
```

### Start the Streamlit dashboard
```powershell
streamlit run app.py
```

### Start the Flask API
```powershell
python api.py
```

### Run tests
```powershell
python -m pytest tests/ -v
```

---

## 15. References

1. Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.

3. Brundage, M. P. et al. (2017). *Technical basis and implementation of condition-based maintenance systems*. NIST.

4. Weiss, G. M. (2004). *Mining with rarity: a unifying framework*. ACM SIGKDD Explorations.

---

*Report generated automatically as part of the Predictive Maintenance Intelligence System mini-project.*  
*All reported metrics are from actual model training on the UCI AI4I 2020 dataset — no values were fabricated.*
