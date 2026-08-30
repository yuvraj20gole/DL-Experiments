# Model Deployment — Flask REST API + Streamlit UI

College deep learning assignment: deploying the trained MNIST baseline ANN from Experiment 2 using a Flask REST API backend and an interactive Streamlit UI frontend.

## Objective

- Serve the pre-trained `baseline_ann.keras` model via a RESTful API built with Flask.
- Expose `/health` and `/predict` endpoints with image preprocessing (grayscale conversion, resizing to 28x28, normalization to [0, 1], flattening to 1x784).
- Provide a user-friendly frontend in Streamlit for uploading digit images and displaying predictions with confidence scores.

## Reused Model Dependency

This project uses the trained baseline ANN model from **Experiment 2** (`DL Experiment 2/baseline-model-tracking/models/baseline_ann.keras`), copied to `models/baseline_ann.keras`.

## Project Structure

```
model-deployment-api-ui/
├── models/
│   └── baseline_ann.keras
├── app.py
├── ui.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

Use the existing native **arm64** Python virtual environment.

### 1. Install Dependencies

```bash
cd "DL Experiment 8/model-deployment-api-ui"
pip install -r requirements.txt
```

### 2. Start Flask Backend API

In Terminal 1:

```bash
python app.py
```

The API starts on `http://127.0.0.1:5002` (or `http://0.0.0.0:5002`).

### 3. Launch Streamlit Frontend

In Terminal 2:

```bash
streamlit run ui.py
```

The UI opens in your browser at `http://localhost:8501`.

---

## Testing the API Directly

### A. Health Check

**Using `curl`:**
```bash
curl http://127.0.0.1:5002/health
```
*Expected response:*
```json
{"status": "ok"}
```

---

### B. Digit Prediction

**Using `curl`:**
```bash
curl -X POST http://127.0.0.1:5002/predict -F "file=@sample_digit.png"
```
*Expected response:*
```json
{
  "predicted_digit": 7,
  "confidence": 0.9998
}
```

**Using Python `requests`:**
```python
import requests

url = "http://127.0.0.1:5002/predict"
with open("sample_digit.png", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
```
