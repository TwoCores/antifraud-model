import os
import logging
import json
from threading import Lock
from fastapi import FastAPI
from fastapi import Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from catboost import CatBoostClassifier, Pool

from constants import FEATURES, CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)


MODEL_PATH = "catboost_fraud_model.cbm"
MODEL_METADATA_PATH = "model_metadata.json"

model = CatBoostClassifier()
model.load_model(MODEL_PATH)

_metadata_lock = Lock()
if os.path.exists(MODEL_METADATA_PATH):
    with open(MODEL_METADATA_PATH, "r") as f:
        model_metadata = json.load(f)
else:
    model_metadata = {}

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
    current_threshold = model_metadata.get("threshold", 0.4)
    block = int(proba >= current_threshold)

    return {
        "fraud_probability": round(proba, 10),
        "block_transaction": bool(block),
        "threshold": current_threshold
    }

@app.get("/model_metadata")
def get_model_metadata():
    with _metadata_lock:
        return dict(model_metadata)


# TODO: for now, it stores in memory only and is lost after restart
@app.post("/set_threshold")
def set_threshold(threshold: float = Query(..., alias="v", description="New threshold value")):
    model_metadata["threshold"] = threshold

    return {
        "threshold": model_metadata["threshold"]
    }
