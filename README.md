# 🛒 SmartShop AI - Intelligent E-commerce Recommendation System

> Sistem rekomendasi ritel berbasis **Hybrid Filtering (Content-Based + Collaborative)** menggunakan **K-Means Clustering** dan **FastAPI**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## 🌟 Fitur Unggulan

* **Smart Segmentation:** Mengelompokkan user secara otomatis menjadi 4 Persona (Newbie, Window Shopper, Loyalist, Sultan) menggunakan algoritma **K-Means**.
* **Secure Auth:** Sistem Login & Register aman dengan **JWT (JSON Web Token)** dan hashing **Bcrypt**.
* **Real-time Prediction:** Rekomendasi produk muncul dalam milidetik berdasarkan input perilaku user (Recency, Frequency, Monetary).
* **Interactive UI:** Dashboard modern berbasis HTML5 + Tailwind CSS (SPA).
* **Data Persistence:** Semua riwayat prediksi tersimpan di PostgreSQL untuk audit.

## 🏗️ Arsitektur Sistem

Project ini menggunakan pola **3-Tier Architecture**:
1.  **Frontend:** HTML5, Tailwind CSS, Vanilla JS (Fetch API).
2.  **Backend:** FastAPI (Python), SQLAlchemy, Pydantic.
3.  **Machine Learning:** Scikit-Learn (K-Means, StandardScaler).
4.  **Database:** PostgreSQL 15.
5.  **Infrastructure:** Docker & Docker Compose.

## 🚀 Cara Menjalankan (Installation)

### Metode 1: Docker (Direkomendasikan)
Pastikan Docker Desktop sudah terinstall.

```bash
# 1. Clone Repository
git clone [https://github.com/username-lu/ecommerce-recommender.git](https://github.com/username-lu/ecommerce-recommender.git)
cd ecommerce-recommender

# 2. Jalankan Aplikasi
docker compose up --build

# 3. Seeding Data Awal (Wajib untuk Database Baru)
docker compose exec web python scripts/3_seed_db.py
