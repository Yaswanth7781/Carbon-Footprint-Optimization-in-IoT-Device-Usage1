import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "iot_energy_large.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


def build_models() -> dict:
    return {
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42,
        ),
        "SVR (RBF)": SVR(kernel="rbf", C=10, gamma="scale"),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10, random_state=42,
        ),
        "KNN (k=5)": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    return mae, rmse, r2


def main():
    print("Loading dataset …")
    df = pd.read_csv(DATA_PATH)
    print(f"   Rows: {len(df)}  |  Columns: {list(df.columns)}\n")

    df["co2"] = df["energy"] / 1000 * 0.82

    le = LabelEncoder()
    df["device_enc"] = le.fit_transform(df["device"])

    feature_cols = ["device_enc", "hour", "day", "temperature",
                    "humidity", "power", "duration"]
    X = df[feature_cols].values
    y = df["energy"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42,
    )

    models = build_models()
    results = []

    print("=" * 72)
    print(f"{'Model':<22s}  {'MAE':>10s}  {'RMSE':>10s}  {'R²':>10s}  {'Time (s)':>9s}")
    print("-" * 72)

    for name, model in models.items():
        t0 = time.time()
        if name == "XGBoost":
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train)
        elapsed = time.time() - t0

        mae, rmse, r2 = evaluate(model, X_test, y_test)
        results.append((name, mae, rmse, r2, elapsed, model))

        print(f"{name:<22s}  {mae:>10.6f}  {rmse:>10.6f}  {r2:>10.6f}  {elapsed:>8.3f}s")

    print("=" * 72)

    results.sort(key=lambda r: r[2])
    best_name, best_mae, best_rmse, best_r2, _, best_model = results[0]

    print(f"\nBest model: {best_name}")
    print(f"   MAE  = {best_mae:.6f}")
    print(f"   RMSE = {best_rmse:.6f}")
    print(f"   R²   = {best_r2:.6f}")

    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(le,         os.path.join(MODEL_DIR, "label_encoder.pkl"))

    print(f"\nSaved best model ({best_name}), scaler, and label-encoder to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
