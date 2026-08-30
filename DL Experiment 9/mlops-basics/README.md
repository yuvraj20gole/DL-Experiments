# MLOps Basics — Containerization (Docker), Automated Testing (Pytest), and CI/CD (GitHub Actions)

College deep learning assignment: establishing essential MLOps practices around the trained MNIST baseline ANN model and Flask API.

## Objective

- **Automated Unit & Integration Testing:** Test `/health` and `/predict` endpoints (valid image upload & missing file error cases) with `pytest`.
- **Containerization:** Package the Flask REST API into a reproducible, lightweight `Dockerfile`.
- **Continuous Integration (CI):** Automate test execution on every commit/PR via GitHub Actions workflow (`.github/workflows/ci.yml`).

## Project Structure

```
mlops-basics/
├── .github/
│   └── workflows/
│       └── ci.yml
├── models/
│   └── baseline_ann.keras
├── tests/
│   └── test_api.py
├── app.py
├── ui.py
├── sample_digit.png
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Local Testing with Pytest

Run the test suite using the existing `arm64` virtual environment:

```bash
cd "DL Experiment 9/mlops-basics"
pytest -v tests/
```

### 2. Local Execution (Flask & Streamlit)

```bash
# Terminal 1: Flask API
python app.py

# Terminal 2: Streamlit UI
streamlit run ui.py
```

### 3. Docker Containerization

```bash
# Build the Docker image
docker build -t dl-lab-app:latest .

# Run the container mapping port 5000
docker run -p 5000:5000 dl-lab-app:latest

# In another terminal, test the containerized service
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict -F "file=@sample_digit.png"
```
