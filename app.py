from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostClassifier, Pool

from constants import FEATURES, CATEGORICAL_FEATURES

MODEL_PATH = "model/catboost_fraud_model.cbm"

model = CatBoostClassifier()
model.load_model(MODEL_PATH)

app = FastAPI(title="Fraud Detection API")

class ModelFeatures(BaseModel):
    cst_dim_id: int
    monthly_os_changes: int = 0
    monthly_phone_model_changes: int = 0
    last_phone_model_categorical: str = ""
    last_os_categorical: str = ""
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

@app.post("/predict")
def predict_fraud(data: ModelFeatures):
    pool = Pool(
        [list(data.model_dump().values())],
        feature_names=FEATURES,
        cat_features=CATEGORICAL_FEATURES
    )

    proba = model.predict_proba(pool)[0][1]
    block = int(proba >= 0.5)

    return {
        "fraud_probability": round(proba, 10),
        "block_transaction": bool(block)
    }
