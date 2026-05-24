import requests
import pandas as pd
import os
from datetime import datetime

# New Delhi coordinates
LATITUDE  = 28.6139
LONGITUDE = 77.2090
CITY      = "New Delhi"

def ingest_data():
    print(f"Fetching historical weather data for {CITY} from OpenMeteo Archive...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  LATITUDE,
        "longitude": LONGITUDE,
        "start_date": "2024-01-01",
        "end_date": "2025-12-31",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure",
            "cloud_cover",
            "dew_point_2m"
        ],
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(url, params=params)
    assert response.status_code == 200, f"API error: {response.status_code}"

    data = response.json()["hourly"]
    df   = pd.DataFrame(data)

    # Convert time column
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour

    # Drop the raw time column
    df = df.drop(columns=["time"])

    # Drop rows with nulls
    df = df.dropna()

    print(f"Fetched {len(df)} hourly records from {df['date'].min()} to {df['date'].max()}")
    print(f"Columns: {df.columns.tolist()}")

    # Create target: did it rain that day? (precipitation > 0.5mm in that hour)
    df["rained"] = (df["precipitation"] > 0.5).astype(int)

    print(f"Rainy hours: {df['rained'].sum()} / {len(df)} total hours")
    print(f"Rain rate: {df['rained'].mean():.2%}")

    # Save
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/weather.csv", index=False)
    print(f"Saved to data/raw/weather.csv")

    return "data/raw/weather.csv"

if __name__ == "__main__":
    ingest_data()