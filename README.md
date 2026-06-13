# 🏭 Steel Surface Defect Detection — AI Inspector

> An AI system that looks at steel surfaces and tells you if there's a defect, what kind it is, and where it is. Think of it like a smart camera for a steel factory.

---

## 🤔 What Does This Do?

Steel factories produce thousands of metal sheets daily. Humans can't inspect every sheet — they get tired, miss things. This AI does it automatically:

1. **You give it a photo** of a steel surface
2. **It tells you** → "This has Scratches" (95% confident)
3. **It draws a box** around the defect
4. **It saves a report** for quality control

---

## 🔍 What Defects Can It Find?

| Defect | What It Looks Like |
|--------|-------------------|
| Crazing | Fine cracks like a shattered phone screen |
| Inclusion | Tiny foreign particles stuck in the steel |
| Patches | Irregular blotchy areas |
| Pitted Surface | Small holes/dents |
| Rolled-in Scale | Flaky oxide pressed into the surface |
| Scratches | Long line marks |

The dataset has **1,800 images** (300 per defect type).

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install

```bash
git clone https://github.com/atulpandey5678/CV_Object_detection.git
cd CV_Object_detection
pip install -r requirements.txt
```

### Step 2: Download Dataset & Prepare

```bash
python scripts/download_dataset.py
python scripts/prepare_dataset.py
```

### Step 3: Run the App

```bash
# Start the web app
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser. Upload a steel image → get results.

---

## 🏋️ Train Your Own Model

```bash
# Train a ResNet50 classifier (takes ~10-30 min depending on GPU)
python scripts/train.py --backbone resnet50 --epochs 50

# Or compare all models
python scripts/train.py --compare-all
```

---

## 🌐 Run the Full System (API + Web App)

```bash
# Terminal 1: Start API server
uvicorn api.main:app --port 8000

# Terminal 2: Start web app
streamlit run frontend/app.py
```

Or use Docker (one command for everything):

```bash
docker-compose up --build
```

This starts: Web App (port 8501) + API (port 8000) + Database + MLflow

---

## 📡 API Endpoints

| What | Endpoint | How |
|------|----------|-----|
| Detect defects | `POST /predict` | Upload an image |
| View history | `GET /history` | See past inspections |
| Get stats | `GET /analytics` | Defect counts & trends |
| Delete record | `DELETE /inspection/{id}` | Remove a record |
| Health check | `GET /health` | Check if server is alive |

Try it: http://localhost:8000/docs (interactive API docs)

---

## 📁 Project Structure (Simplified)

```
├── src/                  # Core AI code
│   ├── models/           # Neural networks (ResNet50, YOLOv8)
│   ├── training/         # Training loops
│   ├── inference/        # Prediction pipeline
│   └── explainability/   # Grad-CAM (why model made a decision)
├── api/                  # REST API (FastAPI)
├── frontend/             # Web app (Streamlit)
├── scripts/              # CLI tools (download, train, prepare)
├── config/               # All settings in one place
├── docker-compose.yml    # One-click deployment
└── requirements.txt      # Python packages needed
```

---

## 🛠️ Tech Stack

- **AI/ML**: PyTorch, YOLOv8, ResNet50, EfficientNet, MobileNet
- **Backend**: FastAPI + PostgreSQL
- **Frontend**: Streamlit + Plotly
- **MLOps**: MLflow (experiment tracking)
- **Deployment**: Docker

---

## 📊 Expected Results

| Model | Accuracy | Speed (GPU) |
|-------|----------|-------------|
| ResNet50 | ~96% | ~50ms/image |
| EfficientNet-B0 | ~95% | ~40ms/image |
| MobileNetV2 | ~93% | ~25ms/image |

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

Services started:
- `http://localhost:8501` — Web App
- `http://localhost:8000` — API
- `http://localhost:5000` — MLflow Dashboard

---

## 📝 License

Educational & research use. Dataset credit: NEU Surface Defect Database (Northeastern University, China).

---

## 👤 Author

**Atul Pandey** — [GitHub](https://github.com/atulpandey5678)
