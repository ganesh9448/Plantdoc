<div align="center">

# 🌿 PlantDoc

### AI-Powered Plant Leaf Disease Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)

**Upload a plant leaf photo → get instant disease diagnosis + treatment advice**

*No login required · No data stored · Completely free · MIT licensed*

</div>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Project Structure](#-project-structure)
4. [Quick Start (Local)](#-quick-start-local)
5. [Deploy with Docker](#-deploy-with-docker)
6. [Environment Variables](#-environment-variables)
7. [API Reference](#-api-reference)
8. [Tech Stack](#-tech-stack)
9. [Disease Classes](#-disease-classes)
10. [Troubleshooting](#-troubleshooting)

---

## 🌱 Project Overview

PlantDoc is a B.Tech final-year project that demonstrates end-to-end deployment of a deep learning system for precision agriculture. It combines:

- **Two-stage AI pipeline** — a YOLOv8 gatekeeper validates input images before the CNN disease model runs
- **Generative AI recommendations** — Gemini 1.5 Flash produces tailored treatment advice
- **Multi-language UI** — English, Telugu, Hindi, Malayalam, Tamil
- **Production-ready backend** — FastAPI with logging, health checks, and input validation

---

## 🏗 Architecture

```
User uploads leaf image
        │
        ▼
┌─────────────────────┐
│  Frontend (HTML/JS) │  dashboard.html + upload.js
└──────────┬──────────┘
           │ POST /predict
           ▼
┌─────────────────────────────────────────┐
│           FastAPI Backend               │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Step 1: YOLO Leaf Gatekeeper   │   │
│  │  model/leaf_detector.pt         │   │
│  │  → leaf (conf ≥ 60%) → proceed  │   │
│  │  → non-leaf            → HTTP 400│  │
│  └────────────────┬────────────────┘   │
│                   │                     │
│  ┌────────────────▼────────────────┐   │
│  │  Step 2: CNN Disease Classifier │   │
│  │  model/leaf_disease_model.h5    │   │
│  │  → 38+ disease classes          │   │
│  └────────────────┬────────────────┘   │
│                   │                     │
│  ┌────────────────▼────────────────┐   │
│  │  POST /recommendations          │   │
│  │  Gemini 1.5 Flash (if key set)  │   │
│  │  → fallback: static text        │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
plantdoc/
│
├── backend/                    # Python FastAPI backend
│   ├── app.py                  # Main application — routes, models, startup
│   ├── classes.txt             # 38 disease class names (one per line)
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # Environment variable template
│   ├── Dockerfile              # Docker image for the backend
│   │
│   ├── model/                  # AI model files (not in Git — see below)
│   │   ├── leaf_disease_model.h5   # TensorFlow CNN (disease classifier)
│   │   └── leaf_detector.pt        # YOLOv8 binary classifier (leaf gatekeeper)
│   │
│   └── utils/
│       ├── __init__.py
│       └── image_utils.py      # Image resize + normalisation helper
│
├── frontend/                   # Static HTML/CSS/JS frontend
│   ├── dashboard.html          # Main single-page app
│   ├── index.html              # Redirect to dashboard.html
│   │
│   ├── css/
│   │   └── style.css           # Full botanical dark-theme stylesheet
│   │
│   └── js/
│       ├── i18n.js             # Multi-language translations (EN/TE/HI/ML/TA)
│       ├── api.js              # Backend HTTP client (predict + recommendations)
│       ├── dashboard.js        # Navigation, language selector, toast system
│       └── upload.js           # Upload UX, result rendering, error handling
│
├── docker-compose.yml          # One-command deploy (backend + nginx frontend)
├── nginx.conf                  # Nginx config for static frontend serving
├── .gitignore
├── .dockerignore
└── README.md                   # This file
```

---

## ⚡ Quick Start (Local)

### Prerequisites

- Python 3.10+
- pip

### 1. Clone / unzip the project

```bash
cd plantdoc/backend
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (optional)
```

### 5. Start the backend

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API is now live at **http://127.0.0.1:8000**
Interactive docs: **http://127.0.0.1:8000/docs**

### 6. Open the frontend

Open `frontend/dashboard.html` in any modern browser — no server needed.

---

## 🐳 Deploy with Docker

Deploy the full stack (backend + nginx frontend) with a single command:

```bash
# 1. Copy and configure the environment file
cp backend/.env.example backend/.env
nano backend/.env   # add your GEMINI_API_KEY

# 2. Build and start both services
docker-compose up --build -d

# 3. View logs
docker-compose logs -f

# 4. Stop
docker-compose down
```

| Service  | URL                        |
|----------|----------------------------|
| Frontend | http://localhost            |
| Backend  | http://localhost:8000       |
| API Docs | http://localhost:8000/docs  |

---

## 🔧 Environment Variables

Create `backend/.env` from `backend/.env.example`:

| Variable                   | Default                    | Description                                      |
|----------------------------|----------------------------|--------------------------------------------------|
| `GEMINI_API_KEY`           | *(empty)*                  | Gemini AI key for dynamic recommendations        |
| `YOLO_MODEL_PATH`          | `model/leaf_detector.pt`   | Path to YOLOv8 leaf gatekeeper model             |
| `LEAF_CONFIDENCE_THRESHOLD`| `0.60`                     | Min confidence to accept image as a leaf (0–1)   |
| `MAX_FILE_SIZE_MB`         | `10`                       | Maximum upload file size in MB                   |
| `ALLOWED_ORIGINS`          | `*`                        | CORS allowed origins (comma-separated)           |

> **Security:** Never commit your `.env` file. Use `.env.example` for documentation.

---

## 📡 API Reference

### `GET /`
System info and health check.

### `GET /health`
Lightweight liveness probe (used by Docker health checks).

### `GET /classes`
Returns all 38 disease class names.

### `POST /predict`
Upload a leaf image and get a disease prediction.

**Request:** `multipart/form-data` with field `file` (JPG/PNG/WebP, max 10 MB)

**Response (success):**
```json
{
  "status": "success",
  "disease": "Apple___Apple_scab",
  "confidence": 94.72,
  "leaf_confidence": 98.13
}
```

**Response (non-leaf rejected):**
```json
{
  "status": "rejected",
  "leaf_confidence": 0.12,
  "message": "❌ This does not appear to be a plant leaf image..."
}
```

### `POST /recommendations`
Get treatment recommendations for a disease.

**Request:**
```json
{ "disease_name": "Apple___Apple_scab" }
```

**Response:**
```json
{
  "pesticide":  { "name": "...", "description": "...", "steps": ["..."] },
  "organic":    { "name": "...", "description": "...", "steps": ["..."] },
  "prevention": { "name": "...", "description": "...", "steps": ["..."] },
  "youtube_links": [{ "title": "...", "url": "https://..." }]
}
```

---

## 🛠 Tech Stack

| Layer        | Technology                          |
|--------------|-------------------------------------|
| Backend      | Python 3.11, FastAPI, Uvicorn       |
| Disease CNN  | TensorFlow / Keras (`.h5` model)    |
| Gatekeeper   | YOLOv8-cls via `ultralytics`        |
| Recommendations | Google Gemini 1.5 Flash          |
| Image processing | Pillow, NumPy                   |
| Frontend     | Vanilla HTML5 / CSS3 / JavaScript   |
| i18n         | Custom lightweight i18n module      |
| Container    | Docker, Docker Compose, Nginx       |

---

## 🌿 Disease Classes

The model detects **38 plant disease categories** across 14 plant types:

Apple (scab, black rot, cedar rust, healthy) · Blueberry (healthy) ·
Cherry (powdery mildew, healthy) · Corn (cercospora, common rust, northern blight, healthy) ·
Grape (black rot, esca, leaf blight, healthy) · Orange (huanglongbing) ·
Peach (bacterial spot, healthy) · Pepper (bacterial spot, healthy) ·
Potato (early blight, late blight, healthy) · Raspberry (healthy) ·
Soybean (healthy) · Squash (powdery mildew) · Strawberry (leaf scorch, healthy) ·
Tomato (bacterial spot, early blight, late blight, leaf mold, septoria, spider mites, target spot, mosaic virus, yellow leaf curl, healthy)

---

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| `❌ Disease CNN failed to load` | Ensure `model/leaf_disease_model.h5` exists in `backend/model/` |
| `❌ YOLO model not found` | Ensure `model/leaf_detector.pt` exists in `backend/model/` |
| `Cannot connect to server` | Start backend: `uvicorn app:app --reload` |
| Gemini not working | Add `GEMINI_API_KEY` to `backend/.env` — static fallback is always active |
| `415 Unsupported media type` | Upload JPG, PNG, or WebP only |
| `413 File too large` | Reduce image size below 10 MB |

---

## 📄 License

MIT License — free to use, modify, and distribute.
See [LICENSE](LICENSE) for full text.

---

<div align="center">
Built with ❤️ for precision agriculture · B.Tech Final Year Project
</div>
