# 📄 DocClassifier AI

A full-stack intelligent document classification system. Upload any document — PDF, image, or text — and get an instant AI-powered prediction of its category with confidence score.

> FastAPI backend · React frontend · JWT auth · Fernet encryption · OCR · Stacking ML classifier

---

## 📸 Demo

![DocClassifier UI](demo.png)

---

## ✨ Features

- **Multi-format support** — PDF (`pdfplumber`), images (`easyocr`), plain text
- **ML Classification** — Stacking classifier (TF-IDF + XGBoost + RandomForest + HistGBT)
- **Secure API** — JWT Bearer tokens + Fernet symmetric encryption + bcrypt password hashing
- **Rate limiting** — `slowapi` middleware, configurable per endpoint
- **Async architecture** — `asyncio` + `aiofiles` for concurrent file processing
- **React frontend** — Drag & drop upload, real-time results, dark/light mode
- **CNN image models** — Separate trained models for malaria detection and cat/dog classification

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           React Frontend                │
│  Sidebar · Upload · Results · Profile   │
└──────────────────┬──────────────────────┘
                   │ HTTP (JWT Bearer)
┌──────────────────▼──────────────────────┐
│           FastAPI Backend               │
│  /api/login · /api/analyze · /api/salt  │
├─────────────────────────────────────────┤
│  JWT Auth │ Rate Limiter │ Fernet Enc.  │
├─────────────────────────────────────────┤
│        Text Extraction Layer            │
│   pdfplumber · easyocr · plain text     │
├─────────────────────────────────────────┤
│      ML Stacking Classifier             │
│   TF-IDF → XGBoost + RF + HistGBT      │
└─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
doc-classifier/
├── main.py              # FastAPI app + server config
├── analyze_router.py    # POST /api/analyze — file upload & prediction
├── login_router.py      # POST /api/login — auth & JWT
├── auth_jwt.py          # JWT create/verify (HS256)
├── chiffrement.py       # Fernet encryption manager
├── database.py          # User storage
├── main_model.py        # ML model loader
├── model.py             # Stacking classifier
├── train.py             # Training script
├── dataset.py           # Dataset loading & preprocessing
├── utils.py             # File handling utilities
├── limiter.py           # Rate limiting setup
├── config.py            # App configuration
├── run_api.py           # Production runner
├── models_images/       # CNN models (malaria, cat/dog)
└── FRONT_END_REACT/            # React app(build)
```

---

## 🔐 Security

| Feature | Implementation |
|---------|---------------|
| Authentication | JWT HS256 with expiry + nbf |
| Password storage | bcrypt hashing + salt |
| Data encryption | Fernet symmetric encryption |
| Rate limiting | slowapi (configurable req/min) |
| CORS | Restricted to frontend origin |

---

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/hounsoubenny-cyber/doc-classifier.git
cd doc-classifier
pip install -r requirements.txt
```

### 2. Configure environment

```bash
nano .env
# Edit .env with your sqlite database uri
```

### 3. Train the model

```bash
python train.py
```

### 4. Start the API

```bash
python run_api.py
# API running at http://localhost:9000
# Swagger docs at http://localhost:9000/api/docs
```

### 5. Start the frontend

```bash
cd frontend
npm install && npm start
# React app at http://localhost:3000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/login` | Register / login, returns JWT | ❌ |
| `GET` | `/salt` | Get bcrypt salt | ❌ |
| `POST` | `/analyze` | Upload files for classification | ✅ JWT |
| `GET` | `/health` | User info & server status | ✅ JWT |
| `GET` | `/api/docs` | Swagger UI | ❌ |

### Example: Classify a document

```bash
curl -X POST http://localhost:9000/api/analyze \
  -H "Authorization: Bearer <token>" \
  -F "files=@document.pdf" \
  -F "username=sam" \
  -F "keep=0"
```

Response:
```json
{
  "results": [
    {
      "filename": "document.pdf",
      "content_type": "application/pdf",
      "predict": {
        "predict": "invoice",
        "proba": 0.94
      },
      "accepted": true
    }
  ]
}
```
---

## 🖼️ CNN Models (Bonus)

The `models_images/` folder contains standalone CNN classifiers:

| Model | Task | Framework |
|-------|------|-----------|
| Malaria detector | Binary classification on cell images | TensorFlow |
| Cat/Dog classifier | Binary classification | TensorFlow |

---

## 📦 Requirements

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-jose[cryptography]>=3.3.0
bcrypt>=4.0.0
cryptography>=41.0.0
slowapi>=0.1.9
pdfplumber>=0.10.0
aiofiles>=23.0.0
aiohttp>=3.9.0
nest-asyncio>=1.5.0
Pillow>=10.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
scikit-optimize>=0.9.0
scikeras>=0.12.0
tensorflow>=2.13.0
torch>=2.0.0
torchvision>=0.15.0
easyocr>=1.7.0
opencv-python-headless>=4.8.0
scikit-image>=0.22.0
python-bidi>=0.4.0
Shapely>=2.0.0
pyclipper>=1.3.0
ninja>=1.11.0
imageio>=2.31.0
zstandard>=0.22.0
python-multipart>=0.0.6
python-dotenv>=1.0.0
pyyaml>=6.0
dill
diskcache
sqlmodel
```

---

## 👤 Author

**Samuel Hounsou**
- GitHub: [@hounsoubenny-cyber](https://github.com/hounsoubenny-cyber)
- LinkedIn: [benny-hounsou](https://linkedin.com/in/benny-hounsou-00a267374)
- Email: hounsoubenny@gmail.com

---

⭐ Star this repo if it sparked ideas!
