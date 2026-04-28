# ────────────────────────────────────────────────────────────
#  Carbon Footprint Optimization — K-Means Clustering
# ────────────────────────────────────────────────────────────
"""
Cluster IoT readings into usage-efficiency groups using K-Means.
Output labels:  efficient  |  moderate  |  high usage
Saves:  models/kmeans.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "iot_energy_large.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# Human-readable labels (sorted by ascending mean energy)
CLUSTER_LABELS = ["efficient", "moderate", "high usage"]


def main():
    # 1. Load data ────────────────────────────────────────────
    print("📂 Loading dataset …")
    df = pd.read_csv(DATA_PATH)

    # 2. Select clustering features ───────────────────────────
    cluster_features = ["energy", "duration", "hour"]
    X = df[cluster_features].values

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Fit K-Means (k=3) ───────────────────────────────────
    print("🔄 Fitting K-Means (k=3) …")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # 4. Map clusters to labels by mean energy ────────────────
    df["cluster_id"] = kmeans.labels_
    cluster_means = df.groupby("cluster_id")["energy"].mean().sort_values()
    label_map = {cid: label for cid, label in zip(cluster_means.index, CLUSTER_LABELS)}
    df["cluster_label"] = df["cluster_id"].map(label_map)

    print("\n📊 Cluster Summary")
    for cid in sorted(label_map.keys()):
        subset = df[df["cluster_id"] == cid]
        print(
            f"   {label_map[cid]:>11s}  |  "
            f"n={len(subset):>4d}  |  "
            f"avg energy={subset['energy'].mean():.2f} kWh  |  "
            f"avg duration={subset['duration'].mean():.2f} h"
        )

    # 5. Save artefacts ───────────────────────────────────────
    # Bundle kmeans model, scaler, and label_map together
    cluster_bundle = {
        "kmeans": kmeans,
        "scaler": scaler,
        "label_map": label_map,
        "features": cluster_features,
    }
    joblib.dump(cluster_bundle, os.path.join(MODEL_DIR, "kmeans.pkl"))

    print(f"\n✅ Saved cluster bundle to  {MODEL_DIR}/kmeans.pkl")


if __name__ == "__main__":
    main()
