import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from catboost import CatBoostClassifier, Pool
import json
import os

logger = logging.getLogger(__name__)

from constants import FEATURES, CATEGORICAL_FEATURES

MODEL_PATH = "catboost_fraud_model.cbm"
MODEL_METADATA_PATH = "model_metadata.json"

model = CatBoostClassifier()
model.load_model(MODEL_PATH)

app = FastAPI(title="Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelFeatures(BaseModel):
    # cst_dim_id: int
    amount: float
    monthly_os_changes: int = 0
    monthly_phone_model_changes: int = 0
    last_phone_model_categorical: str = ""
    last_os_categorical: str = ""
    direction: str = ""
    logins_last_7_days: int = 0
    logins_last_30_days: int = 0
    login_frequency_7d: float = 0.0
    login_frequency_30d: float = 0.0
    freq_change_7d_vs_mean: float = 0.0
    logins_7d_over_30d_ratio: float = 0.0
    avg_login_interval_30d: float = 0.0
    std_login_interval_30d: float = 0.0
    var_login_interval_30d: float = 0.0
    ewm_login_interval_7d: float = 0.0
    burstiness_login_interval: float = 0.0
    fano_factor_login_interval: float = 0.0
    zscore_avg_login_interval_7d: float = 0.0
    # direction_frequency: float = 0.0
    # direction_fraud_rate: float = 0.0

@app.post("/predict")
def predict_fraud(data: ModelFeatures):
    logger.error(f"Received data for prediction: {data}")
    pool = Pool(
        [list(data.model_dump().values())],
        feature_names=FEATURES,
        cat_features=CATEGORICAL_FEATURES
    )

    proba = model.predict_proba(pool)[0][1]
    block = int(proba >= 0.4)

    return {
        "fraud_probability": round(proba, 10),
        "block_transaction": bool(block)
    }

@app.get("/model_metadata")
def get_model_metadata():
    if not os.path.exists(MODEL_METADATA_PATH):
        return {"error": f"{MODEL_METADATA_PATH} not found. Please train the model first."}

    with open(MODEL_METADATA_PATH, "r") as f:
        model_metadata = json.load(f)

    return model_metadata
