from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
import numpy as np
import json
import os
from app.schemas.recommend import PredictionRequest

router = APIRouter(prefix="/cluster", tags=["Cluster"])

# Path Absolut Docker
BASE_DIR = "/code/app/ml"
METRICS_FILE = f"{BASE_DIR}/model_metrics.json"

try:
    scaler = joblib.load(f"{BASE_DIR}/scaler_preproc.joblib")
    kmeans = joblib.load(f"{BASE_DIR}/kmeans_k2.joblib")
except Exception as e:
    print(f"Warning: Failed to load models in cluster.py: {e}")
    scaler = None
    kmeans = None

@router.post("/predict")
def predict_cluster(data: PredictionRequest):
    # ... (Logic prediksi sederhana, opsional jika sudah pakai recommend.py)
    if not scaler or not kmeans:
        raise HTTPException(status_code=500, detail="Model ML belum siap.")
    input_data = data.dict()
    df = pd.DataFrame([input_data])
    df["Monetary_Log"] = np.log1p(df["Monetary"])
    use_cols = ["Recency","Frequency","Monetary_Log","Avg_Items","Unique_Products","Wishlist_Count","Add_to_Cart_Count","Page_Views"]
    try:
        X = scaler.transform(df[use_cols])
        cluster = int(kmeans.predict(X)[0])
        return {"cluster": cluster}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT METRICS WAJIB ADA DI SINI ---
@router.get("/metrics")
def get_model_metrics():
    if not os.path.exists(METRICS_FILE):
        return {"error": "Metrics file not found", "path": METRICS_FILE}
    
    try:
        with open(METRICS_FILE, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"error": f"Failed to read metrics: {str(e)}"}