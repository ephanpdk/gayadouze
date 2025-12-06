from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.encoders import jsonable_encoder
import joblib
import pandas as pd
import numpy as np
import json
import os
import random
import math
from app.database import get_db
from app.models.product import Product
from app.models.log import PredictionLog
from app.models.user import User
from app.schemas.recommend import PredictionRequest
from app.routers.auth import get_current_user

router = APIRouter(prefix="/recommend", tags=["Recommendation"])

BASE_DIR = "/code/app/ml"
METRICS_FILE = f"{BASE_DIR}/model_metrics.json"

models = {"scaler": None, "kmeans": None, "topN": {}, "meta": {}}

def load_models():
    try:
        models["scaler"] = joblib.load(f"{BASE_DIR}/scaler_preproc.joblib")
        models["kmeans"] = joblib.load(f"{BASE_DIR}/kmeans_k2.joblib")
        models["topN"] = joblib.load(f"{BASE_DIR}/topN_by_cluster.joblib")
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "r") as f:
                models["meta"] = json.load(f)
    except Exception as e:
        print(f"Warning load: {e}")

load_models()

@router.post("/user")
def recommend_user(data: PredictionRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # 1. Safety Check
    if not models["scaler"] or not models["kmeans"]:
        load_models()
        if not models["scaler"]:
            raise HTTPException(status_code=503, detail="Model Loading...")

    try:
        # 2. Prepare Data
        input_data = data.dict()
        df = pd.DataFrame([input_data])
        df["Monetary_Log"] = np.log1p(df["Monetary"])
        
        use_cols = ["Recency", "Frequency", "Monetary_Log", "Avg_Items", "Unique_Products", "Wishlist_Count", "Add_to_Cart_Count", "Page_Views"]
        
        # 3. Predict
        X_scaled = models["scaler"].transform(df[use_cols])
        cluster = int(models["kmeans"].predict(X_scaled)[0])
        
        # 4. Distance Calculation
        centroids = models["kmeans"].cluster_centers_
        distances = []
        for i, center in enumerate(centroids):
            dist = float(np.linalg.norm(X_scaled[0] - center))
            distances.append({"cluster": i, "distance": round(dist, 4)})
        
        distances.sort(key=lambda x: x['distance'])
        nearest = distances[0]
        second = distances[1] if len(distances) > 1 else None
        
        # Confidence
        margin = (second['distance'] - nearest['distance']) if second else 0
        confidence = min(100, max(50, (margin * 50) + 50))

        # 5. BUSINESS LOGIC & EXPLAINABILITY (SAFE MODE)
        # Bagian ini yang sering bikin error, kita bungkus safe logic
        readable_cols = models["meta"].get("feature_readable", use_cols)
        z_scores = X_scaled[0]
        drivers = []
        
        for i, val in enumerate(z_scores):
            score = float(val)
            if abs(score) < 0.8: continue
            drivers.append({
                "feature": readable_cols[i] if i < len(readable_cols) else use_cols[i],
                "score": round(score, 2),
                "description": "High" if score > 0 else "Low",
                "sentiment": "positive" if score > 0 else "negative",
                "impact": abs(score)
            })
        drivers.sort(key=lambda x: x['impact'], reverse=True)

        # Generate Text Explanations
        cluster_names = ["Newbie", "Window Shopper", "Loyalist", "Sultan"]
        current_name = cluster_names[cluster]
        
        why_text = f"User classified as {current_name} based on patterns in {drivers[0]['feature'] if drivers else 'general behavior'}."
        compare_text = f"Closest alternative profile is {cluster_names[second['cluster']]}." if second else "Distinct usage profile."
        anomaly_text = "No significant anomalies detected in transaction vector."
        
        if drivers and drivers[0]['impact'] > 2.5:
            anomaly_text = f"Anomaly: {drivers[0]['feature']} is exceptionally {'high' if drivers[0]['score']>0 else 'low'} (Z={drivers[0]['score']})."

        # 6. Recommendations
        raw_recs = models["topN"].get(cluster, [])
        product_ids = [item['product_id'] for item in raw_recs] if isinstance(raw_recs, list) and len(raw_recs) > 0 and isinstance(raw_recs[0], dict) else []
        
        final_recs = []
        if product_ids:
            db_products = db.query(Product).filter(Product.product_id.in_(product_ids)).all()
            base_price = {0: 20, 1: 50, 2: 150, 3: 800}.get(cluster, 50)
            
            for prod in db_products:
                random.seed(prod.product_id)
                final_recs.append({
                    "product_id": prod.product_id,
                    "name": prod.name,
                    "category": prod.category,
                    "price": round(base_price * random.uniform(0.8, 1.2), 2),
                    "rating": round(random.uniform(4.0, 5.0), 1)
                })

        if not final_recs:
             final_recs = [{"product_id": 0, "name": "Item", "category": "General", "price": 0, "rating": 0}]

        recs_clean = jsonable_encoder(final_recs)

        # Log
        try:
            db.add(PredictionLog(user_id=current_user.user_id, predicted_cluster=cluster, recommended_items=recs_clean))
            db.commit()
        except: db.rollback()

        return {
            "cluster": cluster,
            "metrics": {
                "confidence_score": round(confidence, 1),
                "distance_to_centroid": nearest['distance'],
                "feature_drivers": drivers[:3],
                "explanations": {           # <--- DATA BARU YANG DIHARAPKAN UI
                    "why": why_text,
                    "compare": compare_text,
                    "anomaly": anomaly_text
                }
            },
            "recommendations": recs_clean
        }

    except Exception as e:
        print(f"Backend Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))