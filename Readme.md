# 🍲 Recipe Recommendation Web App

Proyek ini terdiri dari dua komponen utama:

1. **Backend Node.js** (folder: `node`) — untuk mengelola permintaan dari web.
2. **Model Machine Learning (Python)** (folder: `python`) — sistem rekomendasi resep berbasis ML.

Kedua bagian ini bekerja bersama untuk menyajikan rekomendasi resep ke website frontend.

---

## 📁 Struktur Folder

```
.
├── node/     ← Backend Node.js
└── python/   ← Model ML dengan Flask API
```

---

## 🧪 Menjalankan Proyek

Ikuti langkah-langkah berikut untuk menjalankan kedua komponen.

### 1. Menjalankan Model Machine Learning (Python)

```bash
cd python
python -m venv venv
```

> **Windows**

```bash
.\venv\Scripts\activate
```

> **Linux/macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Buat file `.env` di dalam folder `python`:

```env
FLASK_PORT=5000
FLASK_ENV=development
```

Jalankan Flask API:

```bash
python app.py
```

---

### 2. Menjalankan Backend Node.js

Buka terminal baru, lalu:

```bash
cd node
npm install
```

Buat file `.env` di dalam folder `node`:

```env
PORT=3000
MONGO_URI=DB
JWT_SECRET=JWT Secret
ML_API_BASE_URL=Model
FLASK_API_URL=http://localhost:5000
```

> Gantilah `MONGO_URI`, `JWT_SECRET`, dan `ML_API_BASE_URL` sesuai kebutuhan Anda.

Jalankan server Node.js:

```bash
npm start
```

---

## ✅ Catatan

* Pastikan Anda menjalankan **Flask API (Python)** terlebih dahulu sebelum memulai **Node.js backend**.
* Keduanya harus berjalan bersamaan agar sistem rekomendasi bekerja optimal.

---

## 🛠 Tech Stack

* **Backend**: Node.js, Express, MongoDB
* **ML Model API**: Python, Flask
* **Rekomendasi**: Collaborative Filtering & Text-based TF-IDF
