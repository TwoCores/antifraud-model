import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import IsolationForest

from constants import FEATURES, CATEGORICAL_FEATURES, IFOREST_FEATURES, TARGET

DATASET_PATH = "clients_patterns_dataset.csv"
MODEL_PATH = "catboost_fraud_model.cbm"

def main():
    df = pd.read_csv(DATASET_PATH)

    numeric_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        if col in ("var_login_interval_30d"):
            df[col] = df[col].str.replace(",", ".").astype(float)

    simulate_target(df)

    X = df.drop(columns=[TARGET, "transid", "transdate"])
    y = df[TARGET]

    for cat_col in CATEGORICAL_FEATURES:
        X[cat_col] = X[cat_col].fillna("missing")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_pool = Pool(X_train, y_train, cat_features=CATEGORICAL_FEATURES)
    test_pool = Pool(X_test, y_test, cat_features=CATEGORICAL_FEATURES)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        class_weights=[1, 10],
        verbose=100
    )

    model.fit(train_pool, eval_set=test_pool)

    preds = model.predict(test_pool)
    preds_proba = model.predict_proba(test_pool)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, preds))

    print("ROC-AUC:", roc_auc_score(y_test, preds_proba))

    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def simulate_target(df):
    clf = IsolationForest(contamination=0.05)
    df[TARGET] = (clf.fit_predict(df[IFOREST_FEATURES]) == -1).astype(int)

if __name__ == "__main__":
    main()