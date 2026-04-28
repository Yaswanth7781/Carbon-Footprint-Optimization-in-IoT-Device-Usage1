# 🌱 Carbon Footprint Optimization System

A full-stack, ML-powered system that **predicts energy usage**, **calculates carbon footprint**, **clusters usage patterns**, and **suggests actionable optimisations** — all served via a **FastAPI** backend and visualised through an interactive **Streamlit** dashboard.

---

## 📑 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#1-clone--install-dependencies)
  - [Train Models](#2-train-the-ml-models)
  - [Start Backend](#3-start-the-fastapi-backend)
  - [Launch Dashboard](#4-launch-the-streamlit-dashboard)
- [API Reference](#-api-reference)
  - [Health Check](#get-)
  - [Predict Energy & CO₂](#post-predict)
  - [Optimisation Suggestions](#post-optimize)
  - [Cluster Analysis](#post-cluster)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
  - [Energy Prediction Model](#1-energy-prediction-model-mltrain)
  - [Usage Clustering Model](#2-usage-clustering-model-mlcluster)
- [Dashboard](#-dashboard)
- [Utility Helpers](#-utility-helpers)
- [Configuration & Constants](#-configuration--constants)
- [Sample Walkthrough](#-sample-walkthrough)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## ✨ Features

| Feature                          | Description                                                                    |
|----------------------------------|--------------------------------------------------------------------------------|
| 🔮 Energy Prediction            | Predicts energy consumption (kWh) using an XGBoost regressor                   |
| 🌫️ CO₂ Estimation              | Converts predicted energy to estimated carbon emissions (kg CO₂)               |
| 📊 Multi-Model Comparison       | Benchmarks 7 regression models and auto-selects the best performer             |
| 🏷️ Usage Clustering             | Groups usage patterns into **efficient**, **moderate**, or **high usage**       |
| 💡 Optimisation Suggestions     | Provides actionable tips based on duration, time-of-day, and energy thresholds |
| 🖥️ Interactive Dashboard        | Streamlit UI with sliders, metrics, and colour-coded cluster badges            |
| ⚡ RESTful API                  | FastAPI backend with automatic Swagger/OpenAPI docs at `/docs`                 |

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
│   IoT Data   │────▶│  ML Training │────▶│   Saved Models (.pkl)│
│   (CSV)      │     │  (train.py   │     │   ├── model.pkl      │
└──────────────┘     │   cluster.py)│     │   ├── scaler.pkl     │
                     └──────────────┘     │   ├── label_encoder  │
                                          │   └── kmeans.pkl     │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │   FastAPI Backend     │
                                          │   (backend/main.py)  │
                                          │   /predict /optimize │
                                          │   /cluster           │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  Streamlit Dashboard  │
                                          │  (dashboard/app.py)  │
                                          └──────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer          | Technologies                                                |
|----------------|-------------------------------------------------------------|
| **ML / Data**  | Python, XGBoost, scikit-learn, pandas, NumPy                |
| **Backend**    | FastAPI, Uvicorn, Pydantic                                  |
| **Dashboard**  | Streamlit, Requests                                         |
| **Serialisation** | joblib                                                   |
| **Data Format**| CSV, XLSX                                                   |

---

## 📁 Project Structure

```
carbon_project/
├── data/
│   ├── iot_energy_large.csv           # Primary training dataset
│   └── iot_ml_training_dataset.xlsx   # Supplementary dataset
├── models/
│   ├── model.pkl                      # Best-performing regression model (XGBoost)
│   ├── scaler.pkl                     # Fitted StandardScaler for features
│   ├── label_encoder.pkl             # Fitted LabelEncoder for device names
│   └── kmeans.pkl                     # K-Means cluster bundle (model + scaler + label map)
├── ml/
│   ├── __init__.py
│   ├── train.py                       # Multi-model comparison & training pipeline
│   └── cluster.py                     # K-Means clustering pipeline
├── backend/
│   └── main.py                        # FastAPI application with 4 endpoints
├── dashboard/
│   └── app.py                         # Streamlit interactive dashboard
├── utils/
│   ├── __init__.py
│   └── helper.py                      # CO₂ calculation & optimisation suggestions
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)

### 1. Clone & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/carbon_project.git
cd carbon_project

# (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install all dependencies
pip install -r requirements.txt
```

### 2. Train the ML Models

Train the regression model and clustering model before using the API:

```bash
# Train & compare 7 regression models — saves the best one
python ml/train.py

# Train K-Means clustering model
python ml/cluster.py
```

> **Note:** This generates all `.pkl` files inside the `models/` directory.

### 3. Start the FastAPI Backend

```bash
python -m uvicorn backend.main:app --reload
```

The API will be available at **http://127.0.0.1:8000**.  
Interactive API docs: **http://127.0.0.1:8000/docs**

### 4. Launch the Streamlit Dashboard

```bash
python -m streamlit run dashboard/app.py --server.headless true
```

The dashboard will open in your browser at **http://localhost:8501**.

> ⚠️ Make sure the FastAPI backend is running **before** launching the dashboard.

---

## 📡 API Reference

**Base URL:** `http://127.0.0.1:8000`

### `GET /`

Health check endpoint.

**Response:**
```json
{
  "message": "Carbon Footprint Optimization API is running 🌿"
}
```

---

### `POST /predict`

Predict energy consumption (kWh) and estimate CO₂ emissions (kg).

**Request Body:**
```json
{
  "device": "Smart Thermostat",
  "hour": 19,
  "day": 15,
  "temperature": 35.0,
  "humidity": 60.0,
  "power": 1500.0,
  "duration": 3.0
}
```

**Response:**
```json
{
  "predicted_energy_kwh": 4.2135,
  "estimated_co2_kg": 0.003455
}
```

**Error (unknown device):**
```json
{
  "error": "Unknown device 'AC'. Known devices: ['Agricultural Sensor', 'EV Charger', ...]"
}
```

---

### `POST /optimize`

Get optimisation suggestions based on usage parameters.

**Request Body:** Same schema as `/predict`.

**Response:**
```json
{
  "suggestions": [
    "Shift usage to off-peak hours (before 6 PM or after 11 PM) to lower grid carbon intensity."
  ]
}
```

---

### `POST /cluster`

Determine the usage-efficiency cluster for given device parameters.

**Request Body:** Same schema as `/predict`.

**Response:**
```json
{
  "cluster_id": 1,
  "cluster_label": "moderate",
  "estimated_energy_kwh": 4.2135
}
```

---

### Request Schema

All `POST` endpoints share the same request body schema:

| Field         | Type    | Description                              | Example              |
|---------------|---------|------------------------------------------|----------------------|
| `device`      | string  | Device name (must match training data)   | `"Smart Thermostat"` |
| `hour`        | int     | Hour of day (0–23)                       | `19`                 |
| `day`         | int     | Day of month (1–31)                      | `15`                 |
| `temperature` | float   | Ambient temperature (°C)                 | `35.0`               |
| `humidity`    | float   | Relative humidity (%)                    | `60.0`               |
| `power`       | float   | Device power rating (W)                  | `1500.0`             |
| `duration`    | float   | Usage duration (hours)                   | `3.0`                |

### Supported Devices

| Device               |
|----------------------|
| Agricultural Sensor  |
| EV Charger           |
| Edge Gateway         |
| Health Monitor       |
| Industrial Sensor    |
| Smart Camera         |
| Smart Lighting       |
| Smart Meter          |
| Smart Thermostat     |
| Wearable Device      |

---

## 🧠 Machine Learning Pipeline

### 1. Energy Prediction Model (`ml/train.py`)

Compares **7 regression models** side-by-side and automatically selects the best performer based on lowest RMSE:

| #  | Model                      | Notes                                             |
|----|----------------------------|---------------------------------------------------|
| 1  | **XGBoost Regressor**      | 300 estimators, max depth 6, learning rate 0.1     |
| 2  | Random Forest Regressor    | 200 estimators, max depth 10                       |
| 3  | Gradient Boosting Regressor| 200 estimators, max depth 5                        |
| 4  | SVR (RBF kernel)           | C=10, gamma=scale                                  |
| 5  | Linear Regression          | Baseline model                                     |
| 6  | Decision Tree Regressor    | Max depth 10                                       |
| 7  | KNN Regressor (k=5)        | Distance-based approach                            |

**Features used (7):**

| Feature       | Description                          |
|---------------|--------------------------------------|
| `device_enc`  | Label-encoded device name            |
| `hour`        | Hour of day (0–23)                   |
| `day`         | Day of month (1–31)                  |
| `temperature` | Ambient temperature (°C)             |
| `humidity`    | Relative humidity (%)                |
| `power`       | Device power rating (W)             |
| `duration`    | Usage duration (hours)              |

**Target variable:** `energy` (kWh)

**Dataset:** 2,000 rows × 8 columns — Train/Test split: 80/20

**Evaluation metrics:** MAE, RMSE, R²

#### Model Comparison Results

| Rank | Model               | MAE       | RMSE      | R² Score  |
|------|---------------------|-----------|-----------|-----------|
| 🥇 1 | **Gradient Boosting** | 0.001522  | 0.003377  | **0.9971** |
| 🥈 2 | Random Forest        | 0.001820  | 0.004480  | 0.9949    |
| 🥉 3 | Decision Tree        | 0.003289  | 0.008010  | 0.9837    |
| 4    | XGBoost              | 0.004211  | 0.009673  | 0.9762    |
| 5    | KNN (k=5)            | 0.010283  | 0.018257  | 0.9151    |
| 6    | Linear Regression    | 0.019523  | 0.031550  | 0.7466    |
| 7    | SVR (RBF)            | 0.076000  | 0.079095  | -0.5927   |

> 🏆 **Best model: Gradient Boosting** — R² = 0.9971, RMSE = 0.003377, MAE = 0.001522

**Saved artifacts:**
- `models/model.pkl` — Best trained model
- `models/scaler.pkl` — Fitted `StandardScaler`
- `models/label_encoder.pkl` — Fitted `LabelEncoder` for device names

---

### 2. Usage Clustering Model (`ml/cluster.py`)

Groups IoT readings into **3 usage-efficiency clusters** using K-Means:

| Cluster Label  | Meaning                                   |
|----------------|-------------------------------------------|
| 🟢 `efficient` | Low energy, short duration, off-peak      |
| 🟡 `moderate`  | Mid-range energy and duration             |
| 🔴 `high usage` | High energy, long duration, peak hours   |

**Clustering features:** `energy`, `duration`, `hour`

#### Cluster Distribution Results

| Cluster      | Count | Avg Energy (kWh) | Avg Duration (h) |
|--------------|-------|-------------------|-------------------|
| 🟢 Efficient  | 863   | 0.02              | 1.66              |
| 🟡 Moderate   | 999   | 0.02              | 4.23              |
| 🔴 High Usage  | 138   | 0.21              | 5.87              |

**Saved artifact:**
- `models/kmeans.pkl` — Bundle containing K-Means model, scaler, label map, and feature list

---

## 📊 Dashboard

The Streamlit dashboard provides an interactive interface for real-time analysis:

- **Sidebar inputs:** Select device, adjust temperature, humidity, power, duration, hour, and day using sliders
- **Analyse button:** Calls all three API endpoints simultaneously
- **Prediction metrics:** Displays predicted energy (kWh), estimated CO₂ (kg), and cluster label
- **Cluster badge:** Colour-coded feedback:
  - ✅ Green — Efficient
  - ⚠️ Yellow — Moderate
  - 🔴 Red — High usage
- **Optimisation tips:** Numbered actionable suggestions

---

## 🔧 Utility Helpers

Located in `utils/helper.py`:

### `calculate_co2(energy_kwh: float) → float`

Converts energy consumption to estimated CO₂ emissions using the formula:

```
CO₂ (kg) = (energy_kwh / 1000) × EMISSION_FACTOR
```

### `get_optimization_suggestions(duration, hour, energy=None) → list[str]`

Returns context-aware suggestions based on:

| Condition                  | Suggestion                                                     |
|----------------------------|----------------------------------------------------------------|
| Duration > 4 hours         | Reduce device usage duration                                   |
| Hour in 18–22 (peak)       | Shift to off-peak hours                                        |
| Energy > 5.0 kWh           | Use energy-efficient device or reduce load                     |
| No issues detected         | "Usage looks efficient — no changes recommended. 🌿"           |

---

## ⚙️ Configuration & Constants

Defined in `utils/helper.py`:

| Constant                  | Value        | Description                              |
|---------------------------|--------------|------------------------------------------|
| `EMISSION_FACTOR`         | `0.82`       | kg CO₂ per kWh (grid average)            |
| `PEAK_HOURS`              | `18–22`      | Peak electricity demand hours             |
| `HIGH_DURATION_THRESHOLD` | `4` hours    | Triggers duration reduction suggestion    |
| `HIGH_ENERGY_THRESHOLD`   | `5.0` kWh    | Triggers high-energy warning              |

---

## 🎯 Sample Walkthrough

```bash
# Step 1: Train models
python ml/train.py
# Output: Comparison table of 7 models, best model saved

python ml/cluster.py
# Output: Cluster summary with counts and average metrics

# Step 2: Start API
uvicorn backend.main:app --reload
# API running at http://127.0.0.1:8000

# Step 3: Test with cURL
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"device":"Smart Thermostat","hour":19,"day":15,"temperature":35,"humidity":60,"power":1500,"duration":3}'

# Step 4: Launch dashboard
streamlit run dashboard/app.py
# Dashboard at http://localhost:8501
```

---

## 🔮 Future Enhancements

- [ ] Add real-time IoT data ingestion via MQTT/Kafka
- [ ] Implement user authentication and multi-tenant support
- [ ] Add historical trend visualisation and time-series forecasting
- [ ] Deploy with Docker and CI/CD pipeline
- [ ] Integrate renewable energy source data for more accurate emission factors
- [ ] Add device-specific energy profiles and recommendations
- [ ] Export reports as PDF from the dashboard

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with 💚 for a greener planet
</p>
