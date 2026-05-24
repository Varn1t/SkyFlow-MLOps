import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle

def preprocess_data():
    print("Loading raw weather data...")
    df = pd.read_csv("data/raw/weather.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Daily aggregation — collapse hourly into daily features
    print("Aggregating hourly data into daily features...")
    daily = df.groupby("date").agg(
        avg_temp        = ("temperature_2m",       "mean"),
        max_temp        = ("temperature_2m",       "max"),
        min_temp        = ("temperature_2m",       "min"),
        avg_humidity    = ("relative_humidity_2m", "mean"),
        max_humidity    = ("relative_humidity_2m", "max"),
        avg_wind        = ("wind_speed_10m",       "mean"),
        max_wind        = ("wind_speed_10m",       "max"),
        avg_pressure    = ("surface_pressure",     "mean"),
        avg_cloud       = ("cloud_cover",          "mean"),
        avg_dew_point   = ("dew_point_2m",         "mean"),
        total_precip    = ("precipitation",        "sum"),
        rainy_hours     = ("rained",               "sum"),
    ).reset_index()

    # Target: did it rain that day? (at least 1 rainy hour)
    daily["rained_today"] = (daily["rainy_hours"] > 0).astype(int)

    # Add time features
    daily["month"]      = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.dayofyear

    # Drop intermediate columns
    daily = daily.drop(columns=["date", "total_precip", "rainy_hours"])

    print(f"Daily records: {len(daily)}")
    print(f"Rainy days: {daily['rained_today'].sum()} / {len(daily)}")

    # Features and target
    X = daily.drop(columns=["rained_today"])
    y = daily["rained_today"]

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} days | Test: {len(X_test)} days")

    # Save
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv",  index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv",   index=False)

    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Saved processed data and scaler.")
    return "data/processed"

if __name__ == "__main__":
    preprocess_data()