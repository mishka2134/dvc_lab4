from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import yaml
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

if __name__ == "__main__":
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)

    os.makedirs("../models", exist_ok=True)
    train_df = pd.read_csv("../data/train.csv")

    X_train = train_df.drop(columns=[config['target']])
    y_train = train_df[config['target']]

    # Масштабирование
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled = scaler.fit_transform(X_train)
    y_transformed = power_trans.fit_transform(y_train.values.reshape(-1, 1))

    # Обучение модели
    lr = SGDRegressor(random_state=config['random_state'])
    clf = GridSearchCV(lr, config['model_params'], cv=3, n_jobs=4)
    clf.fit(X_scaled, y_transformed.reshape(-1))
    best_model = clf.best_estimator_

    # Сохранение артефактов
    joblib.dump(best_model, "../models/energy_model.joblib")
    joblib.dump(scaler, "../models/scaler.joblib")
    joblib.dump(power_trans, "../models/power_transformer.joblib")

    # Сохранение параметров
    with open("../models/best_params.json", "w") as f:
        json.dump(clf.best_params_, f)

    # Оценка на тренировочных данных
    y_pred = best_model.predict(X_scaled)
    y_pred_inv = power_trans.inverse_transform(y_pred.reshape(-1, 1))

    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_inv)),
        'train_mae': mean_absolute_error(y_train, y_pred_inv),
        'train_r2': r2_score(y_train, y_pred_inv)
    }

    with open("../metrics.json", "w") as f:
        json.dump(metrics, f)