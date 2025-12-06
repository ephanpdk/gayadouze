# 🧠 SmartShop AI - Intelligent Retail Analytics System

> Sistem rekomendasi e-commerce berbasis **Unsupervised Learning (K-Means Clustering)** dengan pendekatan **RFM Analysis** dan **Explainable AI (XAI)**.

![Tech Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20Docker%20%7C%20Scikit--Learn-blue)

## 📋 Overview
Project ini mendemonstrasikan implementasi Machine Learning end-to-end untuk segmentasi pelanggan secara real-time. Tidak hanya mengelompokkan user, sistem ini memberikan **alasan matematis** (Feature Contribution) di balik setiap prediksi.

## 🏗️ Arsitektur Sistem
1.  **Data Ingestion:** Synthetic Transaction Data Generator (~5000 vectors).
2.  **Preprocessing:** Log Transformation (Monetary) & StandardScaler (Z-Score).
3.  **Modeling:** K-Means++ (k=4, determined via Elbow Method).
4.  **Inference API:** FastAPI backend dengan latency <100ms.
5.  **Visualization:** Chart.js untuk Centroid Radar & Distribution analysis.

## 🚀 Fitur Utama
* **Real-time Segmentation:** Prediksi persona user (Newbie, Window Shopper, Loyalist, Sultan).
* **Explainability Layer:** Menjelaskan *kenapa* user masuk cluster tertentu berdasarkan jarak Euclidean dan Z-Score fitur.
* **Dynamic Pricing:** Simulasi harga produk menyesuaikan daya beli segmen user.
* **Secure Access:** JWT Authentication & Session Management.

## 🛠️ Cara Menjalankan (Docker)

Pastikan Docker Desktop sudah berjalan.

```bash
# 1. Clone Repository
git clone [https://github.com/USERNAME/gayadouze.git](https://github.com/USERNAME/gayadouze.git)
cd gayadouze

# 2. Build & Run Container
docker compose up --build

# 3. Seeding Data Awal (Wajib)
docker compose exec web python scripts/3_seed_db.py