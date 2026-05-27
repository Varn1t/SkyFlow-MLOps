import requests
import pandas as pd
import numpy as np
import pickle
import os
import mlflow
from datetime import datetime, timedelta

# New Delhi coordinates
LATITUDE  = 28.6139
LONGITUDE = 77.2090
CITY      = "New Delhi"

def get_tomorrow_forecast():
    print(f"[predict] Fetching tomorrow's weather forecast for {CITY}...")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure",
            "cloud_cover",
            "dew_point_2m"
        ],
        "forecast_days": 2,  # Fetch today and tomorrow
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(url, params=params)
    assert response.status_code == 200, f"API Error: {response.status_code}"
    
    # Process into pandas DataFrame
    data = response.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    
    # Filter for tomorrow
    tomorrow = datetime.now().date() + timedelta(days=1)
    df_tomorrow = df[df["date"] == tomorrow].copy()
    
    if len(df_tomorrow) < 24:
        # Fallback if tomorrow isn't fully in the response yet, just use the last 24 hours of forecast
        df_tomorrow = df.tail(24).copy()
        tomorrow = df_tomorrow["date"].iloc[0]

    # Aggregate into daily features matching preprocess.py
    aggregated = {
        "avg_temp":        [df_tomorrow["temperature_2m"].mean()],
        "max_temp":        [df_tomorrow["temperature_2m"].max()],
        "min_temp":        [df_tomorrow["temperature_2m"].min()],
        "avg_humidity":    [df_tomorrow["relative_humidity_2m"].mean()],
        "max_humidity":    [df_tomorrow["relative_humidity_2m"].max()],
        "avg_wind":        [df_tomorrow["wind_speed_10m"].mean()],
        "max_wind":        [df_tomorrow["wind_speed_10m"].max()],
        "avg_pressure":    [df_tomorrow["surface_pressure"].mean()],
        "avg_cloud":       [df_tomorrow["cloud_cover"].mean()],
        "avg_dew_point":   [df_tomorrow["dew_point_2m"].mean()],
        "month":           [tomorrow.month],
        "day_of_year":     [tomorrow.timetuple().tm_yday]
    }
    
    return pd.DataFrame(aggregated), tomorrow

def predict_tomorrow():
    # 1. Fetch live forecast features
    features_df, target_date = get_tomorrow_forecast()

    # 2. Load pre-fitted Scaler
    scaler_path = "models/scaler.pkl"
    if not os.path.exists(scaler_path):
        print(f"[error] Scaler not found at {scaler_path}. Please run preprocess.py first.")
        return
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Scale the features
    features_scaled = pd.DataFrame(
        scaler.transform(features_df),
        columns=features_df.columns
    )

    # 3. Load latest registered model
    run_id_path = "models/latest_run_id.txt"
    if not os.path.exists(run_id_path):
        print(f"[error] Latest Run ID not found at {run_id_path}. Please train a model first.")
        return

    with open(run_id_path, "r") as f:
        run_id = f.read().strip()

    print(f"[predict] Loading model from run: {run_id}...")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    except Exception as e:
        print(f"[warn] Could not load from MLflow Tracking Server ({e}). Attempting local load...")
        # Fallback to local files in mlartifacts if server is offline
        local_model_path = f"mlartifacts/1/{run_id}/artifacts/model"
        model = mlflow.sklearn.load_model(local_model_path)

    # 4. Predict
    prob = model.predict_proba(features_scaled)[0, 1]
    prediction = int(prob >= 0.5)

    # 5. Output beautiful results
    print("\n" + "="*40)
    print(f" 🌧️  SkyFlow-MLOps Rain Forecast")
    print("="*40)
    print(f" Location:         {CITY}")
    print(f" Forecast Date:    {target_date} (Tomorrow)")
    print(f" Rain Probability: {prob * 100:.1f}%")
    print(f" Prediction:       {'YES (Rain likely) ☔' if prediction == 1 else 'NO (Clear sky) ☀️'}")
    print("="*40 + "\n")

if __name__ == "__main__":
    predict_tomorrow()
