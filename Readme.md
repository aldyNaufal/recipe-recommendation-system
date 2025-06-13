# 🧠 Recipe Recommendation ML API

Ini adalah bagian Machine Learning dari sistem rekomendasi resep makanan berbasis **collaborative learning**, menggunakan pendekatan:

* **User-based Collaborative Filtering**
* **Item-based Collaborative Filtering**

Komponen ini dibangun menggunakan **Python** dan disajikan sebagai **Flask API** yang akan dihubungkan oleh backend (Node.js) untuk melayani permintaan rekomendasi.

---

## 📁 Struktur Folder

```
python/
├── app.py               ← API Flask utama
├── services/            ← Modul pemrosesan dan model rekomendasi
│   ├── recipe_recommender.py  ← Kode utama model collaborative & content-based
├── data/                ← Dataset resep dan user
├── model/               ← File model yang telah disimpan (.pkl, .joblib)
├── notebook/            ← Notebook eksplorasi, training, dan evaluasi model
├── config/              ← Konfigurasi lingkungan (opsional)
├── tests/               ← Unit test
├── logs/                ← Log training atau inference
└── requirements.txt     ← Daftar dependencies
```

---

## 🚀 Cara Menjalankan


1. **Buat virtual environment:**

```bash
python -m venv venv
```

2. **Aktifkan environment:**

> **Windows**

```bash
.\venv\Scripts\activate
```

> **macOS/Linux**

```bash
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Buat file `.env` (opsional):**

```env
FLASK_PORT=5000
FLASK_ENV=development
```

5. **Jalankan API Flask:**

```bash
python app.py
```

---

## 📊 Tentang Model Rekomendasi

Model rekomendasi ini menggunakan pendekatan **hybrid**:

### 🧩 1. Collaborative Filtering

* **User-based**: merekomendasikan resep berdasarkan kesamaan pengguna (user-user similarity).
* **Item-based**: merekomendasikan resep berdasarkan kesamaan antar resep (item-item similarity).

### 📝 2. Content-based Filtering

* Menggunakan **TF-IDF vectorization** dari kombinasi:

  * Judul Resep (`Title Cleaned`)
  * Bahan-bahan (`Ingredients Cleaned`)
  * Langkah-langkah (`Steps Cleaned`)

Fitur-fitur tersebut kemudian digunakan untuk menghitung kemiripan antar resep.

### 🧠 Algoritma & Teknik

* **TF-IDF + Cosine Similarity** untuk kemiripan konten
* **Matrix Factorization (optional)** atau **Nearest Neighbors** untuk collaborative filtering
* **Penyatuan skor** dari collaborative dan content-based dengan bobot tertentu

---

## 📌 Contoh Endpoint

Setelah API berjalan (`http://localhost:5000`), beberapa endpoint yang tersedia:

* `GET /recommend/user/<user_id>`
  → Rekomendasi resep berdasarkan histori user (user-based collaborative filtering)

* `GET /recommend/item/<item_id>`
  → Rekomendasi resep serupa dengan resep tertentu (item-based filtering)

* `POST /recommend/content`
  → Rekomendasi berdasarkan isi resep yang diberikan (TF-IDF content-based)

---

## ✅ Catatan Penggunaan

* Dataset di-*preprocess* dan disimpan dalam folder `data/`
* Model yang telah dilatih disimpan dalam folder `model/`
* API ini **tidak memiliki antarmuka pengguna langsung**, namun akan diakses oleh backend (Node.js)

---

## 🛠 Teknologi yang Digunakan

* **Python 3.x**
* **Flask** untuk API
* **scikit-learn**, **pandas**, **numpy**
* **TF-IDF Vectorizer**, **Cosine Similarity**
* **Joblib/Pickle** untuk serialisasi model
* **Custom class** untuk pemrosesan dan rekomendasi
