import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, fbeta_score, roc_auc_score
import shap

from constants import FEATURES, CATEGORICAL_FEATURES, TARGET


shap.initjs()

MODEL_PATH = "catboost_fraud_model.cbm"

def main():
    patterns_df = pd.read_csv("datasets/clients_patterns_dataset.csv")
    transactions_df = pd.read_csv("datasets/transactions_dataset.csv")

    df = pd.merge(transactions_df, patterns_df, on=["cst_dim_id", "transdate"], how="left")

    fix_data_types(df)

    X = df.drop(columns=[TARGET, "cst_dim_id", "transdate", "transdatetime", "docno"])
    y = df[TARGET]

    for cat_col in CATEGORICAL_FEATURES:
        X[cat_col] = X[cat_col].fillna("missing")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_pool = Pool(X_train, y_train, cat_features=CATEGORICAL_FEATURES)
    test_pool = Pool(X_test, y_test, cat_features=CATEGORICAL_FEATURES)

    iterations = 1000
    learning_rate = 0.03
    depth = 6
    loss_function = "Logloss"
    eval_metric = "AUC"
    random_seed = 30
    class_weights = [1, 15]
    # early_stopping_rounds = 100
    l2_leaf_reg = 5
    border_count = 254
    bagging_temperature = 0.5
    min_data_in_leaf = 20
    verbose = 100

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function=loss_function,
        eval_metric=eval_metric,
        random_seed=random_seed,
        class_weights=class_weights,
        # early_stopping_rounds=100,
        l2_leaf_reg=l2_leaf_reg,
        border_count=border_count,
        bagging_temperature=bagging_temperature,
        min_data_in_leaf=min_data_in_leaf,
        verbose=verbose
    )

    model.fit(train_pool, eval_set=test_pool)

    preds_proba = model.predict_proba(test_pool)[:, 1]
    threshold = 0.4
    preds = (preds_proba >= threshold).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, preds))

    feature_importance = model.get_feature_importance(prettified=True)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    fbeta = fbeta_score(y_test, preds, beta=1)
    roc_auc = roc_auc_score(y_test, preds_proba)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score (F-beta=1): {fbeta:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    model_metadata = {
        "model": "CatBoost",
        "model_type": "CatBoostClassifier",
        "model_name": "catboost_antifraud_model",
        "model_params": {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "loss_function": loss_function,
            "eval_metric": eval_metric,
            "random_seed": random_seed,
            "class_weights": class_weights,
            "l2_leaf_reg": l2_leaf_reg,
            "border_count": border_count,
            "bagging_temperature": bagging_temperature,
            "min_data_in_leaf": min_data_in_leaf,
            "verbose": verbose
        },
        "feature_importance": feature_importance.to_dict(orient="records"),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta,
        "roc_auc": roc_auc,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "created_at": datetime.datetime.now().isoformat(),
    }
    save_model_metadata(model_metadata)

    print("Generating SHAP values and saving summary plot...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    shap.save_html("shap/shap_summary.html", shap.plots.force(shap_values[0, ...]))

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap/shap_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP feature importance plot saved to shap/shap_feature_importance.png")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap/shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved to shap/shap_summary_plot.png")

    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def fix_data_types(df):
    numeric_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in numeric_cols:
        if col in ("var_login_interval_30d"):
            df[col] = df[col].str.replace(",", ".").astype(float)

# TODO: save to json, temporary solution
def save_model_metadata(model_metadata, path="model_metadata.json"):
    try:
        with open(path, "r") as f:
            existing = json.load(f)
        if not isinstance(existing, dict):
            existing = {}
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}
    existing.update(model_metadata)

    with open(path, "w") as f:
        json.dump(existing, f, indent=4)
    print(f"Model metadata updated in {path}")


if __name__ == "__main__":
    main()