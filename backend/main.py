# ────────────────────────────────────────────────────────────
# Carbon Footprint Optimization — FastAPI Backend
# ────────────────────────────────────────────────────────────

import os
import sys
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── resolve project root ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.helper import calculate_co2, get_optimization_suggestions

# ── paths ──
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── load models ──
try:
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    cluster_bundle = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))

    kmeans = cluster_bundle["kmeans"]
    cluster_scaler = cluster_bundle["scaler"]
    cluster_label_map = cluster_bundle["label_map"]

except Exception as e:
    print("❌ Model loading failed:", e)
    raise

# ── FastAPI app ──
app = FastAPI(
    title="Carbon Footprint Optimization API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── request schema ──
class InputData(BaseModel):
    device: str
    hour: int
    day: int
    temperature: float
    humidity: float
    power: float
    duration: float


# ── endpoints ──

@app.get("/")
def home():
    return {"message": "API running ✅"}


@app.post("/predict")
def predict(data: InputData):
    try:
        device_enc = label_encoder.transform([data.device])[0]
    except ValueError:
        return {"error": f"Unknown device '{data.device}'"}

    features = np.array([[ 
        device_enc, data.hour, data.day,
        data.temperature, data.humidity,
        data.power, data.duration
    ]])

    features_scaled = scaler.transform(features)

    predicted_energy = float(model.predict(features_scaled)[0])
    co2 = calculate_co2(predicted_energy)

    return {
        "predicted_energy_kwh": round(predicted_energy, 4),
        "estimated_co2_kg": co2
    }


@app.post("/optimize")
def optimize(data: InputData):
    suggestions = get_optimization_suggestions(
        duration=data.duration,
        hour=data.hour
    )
    return {"suggestions": suggestions}


@app.post("/cluster")
def cluster(data: InputData):

    try:
        device_enc = label_encoder.transform([data.device])[0]
    except ValueError:
        return {"error": f"Unknown device '{data.device}'"}

    features = np.array([[ 
        device_enc, data.hour, data.day,
        data.temperature, data.humidity,
        data.power, data.duration
    ]])

    features_scaled = scaler.transform(features)
    estimated_energy = float(model.predict(features_scaled)[0])

    cluster_input = np.array([[estimated_energy, data.duration, data.hour]])
    cluster_input_scaled = cluster_scaler.transform(cluster_input)

    cluster_id = int(kmeans.predict(cluster_input_scaled)[0])
    cluster_label = cluster_label_map.get(cluster_id, "unknown")

    return {
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "estimated_energy_kwh": round(estimated_energy, 4)
    }