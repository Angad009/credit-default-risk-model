# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel

import os
import json

import numpy as np
import pandas as pd
import joblib


# --- Load model + metadata on startup ---

# current directory of this file (api/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# ../models relative to api/
models_dir = os.path.join(current_dir, "..", "models")

model_path = os.path.join(models_dir, "catboost_credit_model.pkl")
metadata_path = os.path.join(models_dir, "model_metadata.json")

# load trained CatBoost model
model = joblib.load(model_path)

# load metadata: best_threshold + feature_columns
with open(metadata_path, "r") as f:
    metadata = json.load(f)

BEST_THRESHOLD = float(metadata["best_threshold"])
FEATURE_COLUMNS = metadata["feature_columns"]


# --- Request schema: raw UCI-style input features (no "default") ---

class ClientFeatures(BaseModel):
    limit_bal: float
    sex: int
    education: int
    marriage: int
    age: int

    pay_0: int
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int

    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float

    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float


# --- Feature engineering (same logic as in the notebook) ---

def make_feature_row(client: ClientFeatures) -> pd.DataFrame:
    """
    Create a single-row DataFrame with all original + engineered features,
    then keep only FEATURE_COLUMNS (the ones used for training).
    """

    # base row from raw JSON input
    base_df = pd.DataFrame([client.dict()])
    df = base_df.copy()

    # original groups
    bill_cols = [f"bill_amt{i}" for i in range(1, 7)]
    pay_cols = [f"pay_amt{i}" for i in range(1, 7)]
    pay_status_cols = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]

    # --- engineered features ---

    # bills
    df["avg_bill_amt"] = df[bill_cols].mean(axis=1)
    df["bill_amt_std"] = df[bill_cols].std(axis=1)
    df["max_bill_amt"] = df[bill_cols].max(axis=1)

    # payments
    df["avg_pay_amt"] = df[pay_cols].mean(axis=1)
    df["pay_amt_std"] = df[pay_cols].std(axis=1)

    # credit utilisation
    df["credit_utilisation_max"] = df["max_bill_amt"] / df["limit_bal"].replace(0, np.nan)
    df["credit_utilisation_max"] = df["credit_utilisation_max"].fillna(0)

    # payment ratios per month
    pay_ratio_cols = []
    for i in range(1, 7):
        bill_col = f"bill_amt{i}"
        pay_col = f"pay_amt{i}"
        ratio_col = f"pay_ratio_{i}"

        df[ratio_col] = df[pay_col] / df[bill_col].replace(0, np.nan)
        pay_ratio_cols.append(ratio_col)

    df[pay_ratio_cols] = df[pay_ratio_cols].fillna(0)

    df["pay_ratio_mean"] = df[pay_ratio_cols].mean(axis=1)
    df["pay_ratio_min"] = df[pay_ratio_cols].min(axis=1)
    df["pay_ratio_max"] = df[pay_ratio_cols].max(axis=1)

    # late payments
    df["num_months_late"] = (df[pay_status_cols] > 0).sum(axis=1)
    df["num_months_very_late"] = (df[pay_status_cols] >= 2).sum(axis=1)
    df["max_delay"] = df[pay_status_cols].max(axis=1)

    # --- IMPORTANT FIX: make sure every training column exists (e.g. 'id') ---

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            # if feature was present during training but not in request,
            # create it with default value 0
            df[col] = 0

    # keep same columns and order as training
    df = df[FEATURE_COLUMNS]

    return df


# --- FastAPI app and endpoints ---

app = FastAPI(
    title="Credit Default Risk API",
    description="Predict credit card default probability using CatBoost.",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {"message": "Credit Default Risk API is running"}


@app.post("/predict")
def predict_default(client: ClientFeatures):
    """
    Predict probability of default for one client
    and return probability + decision based on BEST_THRESHOLD.
    """
    X_row = make_feature_row(client)

    proba_default = float(model.predict_proba(X_row)[:, 1][0])
    will_default = bool(proba_default >= BEST_THRESHOLD)

    return {
        "default_probability": proba_default,
        "will_default": will_default,
        "threshold_used": BEST_THRESHOLD,
    }
