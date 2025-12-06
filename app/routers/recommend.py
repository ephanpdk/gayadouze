from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.encoders import jsonable_encoder
import joblib
import pandas as pd
import numpy as np
from app.database import get_db
from app.models.product import Product
from app.models.log import PredictionLog
from app.models.user import User
from app.schemas.recommend import PredictionRequest, RecommendationResponse
from app.routers.auth import get_current_user

router = APIRouter(prefix="/recommend", tags=["Recommendation"])

BASE_DIR = "/code/app/ml"

try:
    scaler = joblib.load(f"{BASE_DIR}/scaler_preproc.joblib")
    kmeans = joblib.load(f"{BASE_DIR}/kmeans_k2.joblib")
    topN = joblib.load(f"{BASE_DIR}/topN_by_cluster.joblib")
except Exception as e:
    print(f"Warning: Failed to load models: {e}")
    scaler = None
    kmeans = None
    topN = {}

@router.get("/by_cluster/{cid}")
def recommend_by_cluster(cid: int):
    return {"cluster": cid, "recommendations": topN.get(cid, [])}

@router.post("/user")
def recommend_user(
    data: PredictionRequest, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not scaler or not kmeans:
        raise HTTPException(status_code=500, detail="Model ML belum siap di server.")

    input_data = data.dict()
    df = pd.DataFrame([input_data])
    df["Monetary_Log"] = np.log1p(df["Monetary"])

    use_cols = [
        "Recency", "Frequency", "Monetary_Log", "Avg_Items",
        "Unique_Products", "Wishlist_Count", "Add_to_Cart_Count", "Page_Views"
    ]

    try:
        X_scaled = scaler.transform(df[use_cols])
        
        cluster_prediction = kmeans.predict(X_scaled)[0]
        cluster = int(cluster_prediction)
        
        # Hitung Distance ke Centroid (Fitur Akademik)
        centroids = kmeans.cluster_centers_
        distances = []
        for i, center in enumerate(centroids):
            dist = np.linalg.norm(X_scaled - center)
            distances.append({
                "cluster": i,
                "distance": float(round(dist, 4))
            })
        
        distances.sort(key=lambda x: x['distance'])
        nearest = distances[0]
        second_nearest = distances[1] if len(distances) > 1 else None
        
        # Ambil Rekomendasi
        raw_recs = topN.get(cluster, [])
        product_ids = []

        if raw_recs:
            if isinstance(raw_recs[0], dict):
                 product_ids = [item['product_id'] for item in raw_recs]
            elif isinstance(raw_recs[0], (int, np.integer, float)):
                 product_ids = [int(x) for x in raw_recs]
        
        final_recs = []
        if product_ids:
            db_products = db.query(Product).filter(Product.product_id.in_(product_ids)).all()
            for prod in db_products:
                final_recs.append({
                    "product_id": prod.product_id,
                    "name": prod.name,
                    "category": prod.category,
                    "price": round(np.random.uniform(20, 500), 2)
                })
        
        if not final_recs:
             final_recs = [{"product_id": 0, "name": "Tidak ada rekomendasi", "category": "General", "price": 0}]

        recs_clean = jsonable_encoder(final_recs)

        new_log = PredictionLog(
            user_id=current_user.user_id,
            predicted_cluster=cluster,
            recommended_items=recs_clean
        )
        db.add(new_log)
        db.commit()

        return {
            "cluster": cluster,
            "metrics": {
                "user_distance": nearest['distance'],
                "nearest_cluster": nearest['cluster'],
                "next_nearest_cluster": second_nearest['cluster'] if second_nearest else None,
                "margin": float(round(second_nearest['distance'] - nearest['distance'], 4)) if second_nearest else 0,
                "all_distances": distances
            },
            "recommendations": recs_clean
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))