# Credit Card Default Risk Model (CatBoost + SHAP + FastAPI)

End-to-end credit risk modelling project using the **UCI "Default of Credit Card Clients"** dataset.  
Built as a portfolio project for MSc Data Analytics & graduate data / analytics roles.

---

## 1. Problem

Predict whether a credit card customer will **default next month** and provide:

- A **probability of default (PD)** for each customer  
- A **business-friendly decision rule** based on asymmetric costs  
- **Explainable** model behaviour suitable for risk / credit teams

Target: `default` (1 = default next month, 0 = no default).

---

## 2. Dataset

- Source: UCI Machine Learning Repository – *Default of Credit Card Clients*
- 30,000 customers  
- 24 original features + binary target  
- Includes:
  - Demographics (sex, age, education, marriage)  
  - Credit limit (`LIMIT_BAL`)  
  - 6 months of **repayment status** (`PAY_0`…`PAY_6`)  
  - 6 months of **bill amounts** (`BILL_AMT1`…`BILL_AMT6`)  
  - 6 months of **payment amounts** (`PAY_AMT1`…`PAY_AMT6`)

Raw file in this repo: `data/default of credit card clients.xls`.

---

## 3. Approach

### 3.1 Data prep & split

- Cleaned column names and target.
- Train / validation / test split: **60% / 20% / 20%**, stratified by `default`.
- Checked class imbalance and basic distributions.

### 3.2 Feature engineering

Created richer risk features:

**Payment history**

- `num_months_late` – number of months with any delay  
- `num_months_very_late` – months with delay ≥ 2  
- `max_delay` – worst delinquency code across 6 months  
- Kept raw `pay_0`, `pay_2`, … etc.

**Exposure & utilisation**

- `max_bill_amt` – maximum bill across 6 months  
- `avg_bill_amt` – mean bill  
- `bill_amt_std` – volatility of bills  
- `credit_utilisation_max = max_bill_amt / limit_bal`

**Payment behaviour**

- Monthly ratios: `pay_ratio_i = pay_amt_i / bill_amt_i` (i = 1..6)  
- Aggregates:
  - `pay_ratio_mean`, `pay_ratio_min`, `pay_ratio_max`  
  - `avg_pay_amt`, `pay_amt_std`

The same feature engineering logic is used inside the FastAPI service.

### 3.3 Models

Trained and compared:

- **Logistic Regression** (baseline)
- **LightGBM**
- **CatBoost**  ← final model

Model selection metrics on the validation set:

- **ROC AUC**
- **Average Precision (PR AUC)**

CatBoost with engineered features achieved the best performance (~**0.79 ROC AUC** on validation).

### 3.4 Business-driven threshold optimisation

Instead of using threshold = 0.5, used **cost-based** optimisation:

- False Negative (missed defaulter) cost = **5**
- False Positive (flagged as default but actually good) cost = **1**

Steps:

1. Scanned thresholds from 0.10 to 0.90.  
2. For each threshold, computed confusion matrix on the validation set.  
3. Calculated total cost: `cost_fp * FP + cost_fn * FN`.  
4. Chose the threshold with minimum cost.

Best threshold in this run: **≈ 0.14**  
→ high recall for defaulters, accepting more false positives (risk-averse strategy).

### 3.5 Calibration

For the final CatBoost model:

- Plotted **calibration curve** (reliability diagram) on validation.
- Computed **Brier score**.

The curve is close to the diagonal → probabilities are **reasonably well calibrated**.

### 3.6 Explainability (SHAP)

Used **SHAP** to explain CatBoost:

- **SHAP bar plot** – global feature importance.  
- **Beeswarm plot** – how high/low values push predictions up/down.  
- **Dependency plots** for:
  - `num_months_late`
  - `credit_utilisation_max`
  - `limit_bal`

Key insights:

- Recent repayment status (`pay_0`), number and severity of late months, and `credit_utilisation_max` are the strongest drivers of default.  
- Default risk increases sharply with more late months and high utilisation; higher credit limits are associated with lower risk.

### 3.7 Monitoring / drift check

Simple monitoring using train vs test:

- Created `train_df` and `test_df` with features, `default`, and `cat_proba`.  
- Compared **mean / std** of key features between train and test.  
- Plotted histograms for important features (e.g., `credit_utilisation_max`).  
- Compared **ROC AUC**:

  - Train ROC AUC: **0.8319**  
  - Test ROC AUC: **0.7835**

Small train–test gap and similar distributions → good generalisation, limited drift.

### 3.8 Deployment (FastAPI)

Deployed the final CatBoost model as a **FastAPI** service:

- Saved model + metadata in `models/`:
  - `catboost_credit_model.pkl`
  - `model_metadata.json` (stores `best_threshold` and `feature_columns`)  
- API code in `api/main.py`.

`POST /predict`:

- Accepts raw UCI-style fields: `LIMIT_BAL`, `PAY_*`, `BILL_AMT*`, `PAY_AMT*`, plus demographics.
- Recomputes engineered features.
- Returns:

```json
{
  "default_probability": <float>,
  "will_default": <true/false>,
  "threshold_used": <float>
}
```

---

## 4. Results (Test Set)

Final CatBoost model **with** feature engineering on the held-out test set:

- **ROC AUC:** `0.7835`  
- **Average Precision (PR AUC):** `0.5636`

Using the cost-optimised threshold (~`0.14`) on the test set, confusion matrix:

- **TN:** 2676  
- **FP:** 1997  
- **FN:** 265  
- **TP:** 1062  

This gives:

- High **recall** for defaulters (around ~0.80).  
- Moderate **precision** for defaulters (around ~0.35).  
- Total business cost (with FN cost = 5, FP cost = 1) is similar on validation and test, showing stable behaviour on unseen data.

---

## 5. Project structure

```text
credit-default-risk-model/
├── api/
│   └── main.py                    # FastAPI app (model inference + feature engineering)
├── data/
│   └── default of credit card clients.xls   # UCI dataset
├── models/
│   ├── catboost_credit_model.pkl  # Trained CatBoost model
│   └── model_metadata.json        # Feature list + best threshold
└── notebooks/
    └── 01_eda_and_baseline.ipynb  # Full pipeline: EDA → models → SHAP → evaluation
```

---

## 6. How to run locally

### 6.1 Requirements

You’ll need Python 3.10+ and these packages:

- `numpy`, `pandas`
- `scikit-learn`
- `lightgbm`
- `catboost`
- `matplotlib`, `seaborn`
- `shap`
- `fastapi`, `uvicorn`

Example install:

```bash
pip install numpy pandas scikit-learn lightgbm catboost matplotlib seaborn shap fastapi "uvicorn[standard]"
```

### 6.2 Run the notebook

```bash
jupyter notebook
```

Then:

- Open `notebooks/01_eda_and_baseline.ipynb`
- Run cells top to bottom to reproduce the analysis.

### 6.3 Run the FastAPI service

From the repo root:

```bash
cd api
uvicorn main:app --reload
```

API will be available at: `http://127.0.0.1:8000`

Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

#### Example `/predict` request body

```json
{
  "limit_bal": 20000,
  "sex": 2,
  "education": 2,
  "marriage": 1,
  "age": 35,
  "pay_0": 0,
  "pay_2": 0,
  "pay_3": 0,
  "pay_4": 0,
  "pay_5": 0,
  "pay_6": 0,
  "bill_amt1": 5000,
  "bill_amt2": 4000,
  "bill_amt3": 3000,
  "bill_amt4": 2000,
  "bill_amt5": 1000,
  "bill_amt6": 0,
  "pay_amt1": 5000,
  "pay_amt2": 4000,
  "pay_amt3": 3000,
  "pay_amt4": 2000,
  "pay_amt5": 1000,
  "pay_amt6": 0
}
```

The response includes the default probability and decision based on the optimised threshold.

---

