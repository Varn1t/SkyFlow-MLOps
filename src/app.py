from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import os
import mlflow
from src.predict import get_tomorrow_forecast

app = FastAPI(
    title="SkyFlow-MLOps API",
    description="A production-ready prediction API for New Delhi rainfall using optimized Random Forest models.",
    version="1.0.0"
)

# Input data schema for manual feature predictions
class WeatherInput(BaseModel):
    avg_temp: float = Field(..., example=25.4, description="Average temperature in Celsius")
    max_temp: float = Field(..., example=31.2, description="Maximum temperature in Celsius")
    min_temp: float = Field(..., example=19.8, description="Minimum temperature in Celsius")
    avg_humidity: float = Field(..., example=65.2, description="Average relative humidity (%)")
    max_humidity: float = Field(..., example=85.0, description="Maximum relative humidity (%)")
    avg_wind: float = Field(..., example=12.5, description="Average wind speed (km/h)")
    max_wind: float = Field(..., example=22.1, description="Maximum wind speed (km/h)")
    avg_pressure: float = Field(..., example=1010.5, description="Average surface pressure (hPa)")
    avg_cloud: float = Field(..., example=45.0, description="Average cloud cover (%)")
    avg_dew_point: float = Field(..., example=18.2, description="Average dew point temperature in Celsius")
    month: int = Field(..., ge=1, le=12, example=5, description="Month of the year (1-12)")
    day_of_year: int = Field(..., ge=1, le=366, example=145, description="Day of the year (1-366)")

# Global model and scaler variables loaded on startup
model = None
scaler = None

def load_assets():
    global model, scaler
    # 1. Load scaler
    scaler_path = "models/scaler.pkl"
    if not os.path.exists(scaler_path):
        raise RuntimeError(f"Scaler not found at {scaler_path}. Please run preprocessing first.")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # 2. Load latest model
    run_id_path = "models/latest_run_id.txt"
    if not os.path.exists(run_id_path):
        raise RuntimeError(f"Run ID not found at {run_id_path}. Please train the model first.")
    with open(run_id_path, "r") as f:
        run_id = f.read().strip()

    mlflow.set_tracking_uri("http://localhost:5000")
    try:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    except Exception:
        # Fallback to local files if mlflow server is offline
        local_model_path = f"mlartifacts/1/{run_id}/artifacts/model"
        model = mlflow.sklearn.load_model(local_model_path)

@app.on_event("startup")
def startup_event():
    load_assets()

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serves a premium, interactive glassmorphic MLOps landing dashboard."""
    if model is None or scaler is None:
        return """<html><body><h3>System Unhealthy: Model assets not loaded</h3></body></html>"""
    
    with open("models/latest_run_id.txt", "r") as f:
        run_id = f.read().strip()
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SkyFlow-MLOps API Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }}
        body {{
            background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            overflow-x: hidden;
        }}
        .container {{
            max-width: 900px;
            width: 100%;
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 3rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeIn 0.8s ease-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 2.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 1.5rem;
        }}
        .brand {{ display: flex; align-items: center; gap: 1rem; }}
        .brand-logo {{ font-size: 2.5rem; }}
        .brand-title h1 {{ font-size: 1.8rem; font-weight: 800; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .brand-title p {{ font-size: 0.85rem; color: #94a3b8; margin-top: 0.2rem; }}
        .status-badge {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            color: #34d399;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981;
            animation: pulse 1.8s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
            70% {{ transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }}
            100% {{ transform: scale(0.9); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2.5rem;
        }}
        @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        .card {{
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 16px;
            padding: 1.8rem;
        }}
        .card-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .meta-list {{ display: flex; flex-direction: column; gap: 0.8rem; }}
        .meta-item {{ display: flex; justify-content: space-between; font-size: 0.9rem; }}
        .meta-label {{ color: #94a3b8; }}
        .meta-val {{ color: #f1f5f9; font-weight: 600; font-family: monospace; }}
        .interactive-section {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            justify-content: center;
            height: 100%;
        }}
        .btn {{
            background: linear-gradient(135deg, #0284c7, #4f46e5);
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(79, 70, 229, 0.3);
            text-decoration: none;
            display: inline-block;
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(79, 70, 229, 0.5);
        }}
        .btn-secondary {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 1rem;
            box-shadow: none;
            color: #cbd5e1;
            transition: all 0.2s ease;
        }}
        .btn-secondary:hover {{
            background: rgba(255, 255, 255, 0.1);
            box-shadow: none;
        }}
        .tester-card {{
            grid-column: span 2;
        }}
        @media (max-width: 768px) {{ .tester-card {{ grid-column: span 1; }} }}
        .forecast-display {{
            display: none;
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            animation: slideDown 0.4s ease-out;
        }}
        @keyframes slideDown {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .forecast-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }}
        .forecast-result {{ font-size: 1.4rem; font-weight: 800; display: flex; align-items: center; gap: 0.5rem; }}
        .progress-bar-bg {{ width: 100%; height: 8px; background: rgba(255, 255, 255, 0.05); border-radius: 4px; overflow: hidden; margin-top: 0.5rem; }}
        .progress-bar-fill {{ height: 100%; width: 0%; background: linear-gradient(to right, #38bdf8, #818cf8); transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }}
        .loading {{ display: none; color: #94a3b8; font-size: 0.9rem; align-items: center; gap: 0.5rem; margin-top: 1rem; }}
        .spinner {{ width: 18px; height: 18px; border: 2px solid rgba(255, 255, 255, 0.1); border-top-color: #38bdf8; border-radius: 50%; animation: spin 0.8s infinite linear; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="brand">
                <span class="brand-logo">🌤️</span>
                <div class="brand-title">
                    <h1>SkyFlow-MLOps Engine</h1>
                    <p>Production Prediction API for Rain Forecasting</p>
                </div>
            </div>
            <div class="status-badge">
                <div class="status-dot"></div>
                Operational
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-title">⚙️ System Info</div>
                <div class="meta-list">
                    <div class="meta-item">
                        <span class="meta-label">Service</span>
                        <span class="meta-val">SkyFlow-MLOps API</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Location</span>
                        <span class="meta-val" style="color: #38bdf8;">New Delhi, IN</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">MLflow Run ID</span>
                        <span class="meta-val" style="font-size: 0.75rem; max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{run_id}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Model Architecture</span>
                        <span class="meta-val">Random Forest</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="interactive-section">
                    <a href="/docs" class="btn" target="_blank">🔌 Open Swagger Docs</a>
                    <p style="font-size: 0.8rem; color: #64748b; margin-top: 0.8rem;">Explore interactive API endpoints directly</p>
                </div>
            </div>

            <div class="card tester-card">
                <div class="card-title">🔮 Real-Time Forecast Tester</div>
                <p style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 1.2rem;">Click below to fetch live meteorological data from Open-Meteo for tomorrow and run pipeline inference.</p>
                
                <button onclick="runForecast()" class="btn btn-secondary">⚡ Run Tomorrow's Forecast</button>
                
                <div id="loading" class="loading">
                    <div class="spinner"></div>
                    Fetching live atmospheric readings & executing model pipeline...
                </div>

                <div id="forecast-display" class="forecast-display">
                    <div class="forecast-header">
                        <span style="color: #94a3b8; font-size: 0.9rem;">Forecast Date: <strong id="f-date" style="color: #e2e8f0;"></strong></span>
                        <span style="font-weight: 600; color: #38bdf8;" id="f-prob-text"></span>
                    </div>
                    <div class="forecast-result" id="f-result"></div>
                    <div class="progress-bar-bg">
                        <div id="f-progress" class="progress-bar-fill"></div>
                    </div>
                    <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 1rem; font-style: italic;" id="f-recommendation"></p>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function runForecast() {{
            const btn = document.querySelector('.btn-secondary');
            const loader = document.getElementById('loading');
            const display = document.getElementById('forecast-display');
            
            btn.style.opacity = '0.5';
            btn.disabled = true;
            loader.style.display = 'flex';
            display.style.display = 'none';

            try {{
                const response = await fetch('/predict_tomorrow', {{ method: 'POST' }});
                const data = await response.json();
                
                document.getElementById('f-date').innerText = data.forecast_date;
                const prob = Math.round(data.rain_probability * 100);
                document.getElementById('f-prob-text').innerText = prob + '% Rain Prob';
                
                const resultDiv = document.getElementById('f-result');
                if (data.prediction === 1) {{
                    resultDiv.innerHTML = '<span style="color: #38bdf8;">YES</span> (Rain likely) ☔';
                }} else {{
                    resultDiv.innerHTML = '<span style="color: #fbbf24;">NO</span> (Clear sky) ☀️';
                }}
                
                document.getElementById('f-recommendation').innerText = data.recommendation;
                
                // Show display
                display.style.display = 'block';
                
                // Animate progress bar
                setTimeout(() => {{
                    document.getElementById('f-progress').style.width = prob + '%';
                }}, 100);

            }} catch (err) {{
                alert('Inference pipeline error: ' + err.message);
            }} finally {{
                btn.style.opacity = '1';
                btn.disabled = false;
                loader.style.display = 'none';
            }}
        }}
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

@app.post("/predict")
def predict_weather(payload: WeatherInput):
    """Predict rainfall from custom weather features."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model assets are not initialized yet")

    try:
        # Convert Pydantic model to dict, then DataFrame
        features_df = pd.DataFrame([payload.dict()])
        
        # Scale features
        features_scaled = pd.DataFrame(
            scaler.transform(features_df),
            columns=features_df.columns
        )
        
        # Inference
        prob = float(model.predict_proba(features_scaled)[0, 1])
        prediction = int(prob >= 0.5)
        
        return {
            "prediction": prediction,
            "rain_probability": prob,
            "recommendation": "YES (Rain likely) - Carry an umbrella! ☔" if prediction == 1 else "NO (Clear sky) - Enjoy the day! ☀️"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_tomorrow")
def predict_tomorrow():
    """Automatically fetch tomorrow's live forecast and predict precipitation."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model assets are not initialized yet")

    try:
        # Fetch tomorrow's forecast
        features_df, target_date = get_tomorrow_forecast()
        
        # Scale features
        features_scaled = pd.DataFrame(
            scaler.transform(features_df),
            columns=features_df.columns
        )
        
        # Inference
        prob = float(model.predict_proba(features_scaled)[0, 1])
        prediction = int(prob >= 0.5)
        
        return {
            "forecast_date": str(target_date),
            "prediction": prediction,
            "rain_probability": prob,
            "recommendation": "YES (Rain likely) - Carry an umbrella! ☔" if prediction == 1 else "NO (Clear sky) - Enjoy the day! ☀️"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {str(e)}")
