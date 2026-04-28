# ────────────────────────────────────────────────────────────
#  Carbon Footprint Optimization — Streamlit Dashboard
# ────────────────────────────────────────────────────────────
"""
Interactive dashboard that calls the FastAPI backend and displays
energy predictions, CO₂ estimates, cluster labels, and optimisation tips.
"""

import streamlit as st
import requests

API_URL = "https://carbon-footprint-optimization-in-iot-ty6w.onrender.com"

# ── page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Carbon Footprint Optimizer",
    page_icon="🌿",
    layout="centered",
)

st.title("🌱 Carbon Footprint Optimization Dashboard")
st.markdown(
    "Enter device parameters below to **predict energy usage**, "
    "**estimate CO₂ emissions**, and get **optimisation suggestions**."
)

# ── sidebar inputs ──────────────────────────────────────────
st.sidebar.header("⚙️ Device Parameters")

DEVICES = [
    "Agricultural Sensor", "EV Charger", "Edge Gateway",
    "Health Monitor", "Industrial Sensor", "Smart Camera",
    "Smart Lighting", "Smart Meter", "Smart Thermostat",
    "Wearable Device",
]

device      = st.sidebar.selectbox("Device", DEVICES)
temperature = st.sidebar.slider("Temperature (°C)", 10.0, 50.0, 30.0, 0.5)
humidity    = st.sidebar.slider("Humidity (%)", 20.0, 100.0, 60.0, 1.0)
power       = st.sidebar.slider("Power (W)", 50.0, 5000.0, 1500.0, 50.0)
duration    = st.sidebar.slider("Duration (hours)", 0.5, 12.0, 3.0, 0.5)
hour        = st.sidebar.slider("Hour of day (0-23)", 0, 23, 19)
day         = st.sidebar.slider("Day of month (1-31)", 1, 31, 15)

payload = {
    "device": device,
    "hour": hour,
    "day": day,
    "temperature": temperature,
    "humidity": humidity,
    "power": power,
    "duration": duration,
}

# ── predict button ──────────────────────────────────────────
if st.sidebar.button("🔍 Analyse", use_container_width=True):
    with st.spinner("Contacting API …"):
        try:
            # /predict
            pred_resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            pred_data = pred_resp.json()

            # /optimize
            opt_resp = requests.post(f"{API_URL}/optimize", json=payload, timeout=10)
            opt_data = opt_resp.json()

            # /cluster
            clust_resp = requests.post(f"{API_URL}/cluster", json=payload, timeout=10)
            clust_data = clust_resp.json()

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Could not connect to the FastAPI backend. "
                "Make sure `uvicorn backend.main:app --reload` is running on port 8000."
            )
            st.stop()

    # ── display results ─────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "⚡ Predicted Energy",
        f"{pred_data.get('predicted_energy_kwh', 'N/A')} kWh",
    )
    col2.metric(
        "🌫️ Estimated CO₂",
        f"{pred_data.get('estimated_co2_kg', 'N/A')} kg",
    )
    col3.metric(
        "🏷️ Cluster",
        clust_data.get("cluster_label", "N/A").title(),
    )

    # Cluster colour badge
    label = clust_data.get("cluster_label", "")
    if label == "efficient":
        st.success(f"✅ This usage pattern is **efficient**.")
    elif label == "moderate":
        st.warning(f"⚠️ This usage pattern is **moderate**.")
    elif label == "high usage":
        st.error(f"🔴 This usage pattern is **high usage** — consider optimising.")

    # Suggestions
    st.markdown("---")
    st.subheader("💡 Optimisation Suggestions")
    suggestions = opt_data.get("suggestions", [])
    for i, tip in enumerate(suggestions, 1):
        st.info(f"**{i}.** {tip}")

else:
    st.info("👈 Set device parameters in the sidebar and click **Analyse**.")
