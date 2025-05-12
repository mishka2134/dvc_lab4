import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

if __name__ == "__main__":
    test_df = pd.read_csv("../data/test.csv")
    model = joblib.load("../models/energy_model.joblib")
    scaler = joblib.load("../models/scaler.joblib")
    power_trans = joblib.load("../models/power_transformer.joblib")

    X_test = test_df.drop(columns=['Energy Consumption'])
    y_test = test_df['Energy Consumption']

    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    y_pred_inv = power_trans.inverse_transform(y_pred.reshape(-1, 1))

    metrics = {
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_inv)),
        'test_mae': mean_absolute_error(y_test, y_pred_inv),
        'test_r2': r2_score(y_test, y_pred_inv)
    }

    with open("../metrics.json", "w") as f:
        json.dump(metrics, f)
