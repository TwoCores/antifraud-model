FEATURES = [
    # "cst_dim_id",
    "amount",
    "monthly_os_changes", "monthly_phone_model_changes",
    "last_phone_model_categorical", "last_os_categorical", "direction",
    "logins_last_7_days", "logins_last_30_days",
    "login_frequency_7d", "login_frequency_30d",
    "freq_change_7d_vs_mean", "logins_7d_over_30d_ratio",
    "avg_login_interval_30d", "std_login_interval_30d",
    "var_login_interval_30d", "ewm_login_interval_7d",
    "burstiness_login_interval", "fano_factor_login_interval",
    "zscore_avg_login_interval_7d"
]

CATEGORICAL_FEATURES = [
    "last_phone_model_categorical",
    "last_os_categorical",
    "direction"
]

TARGET = "target"
