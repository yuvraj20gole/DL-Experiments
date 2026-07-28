# MNIST Baseline ANN with Experiment Tracking

## Objective

Build a reusable baseline artificial neural network (ANN) on MNIST and track hyperparameter experiments manually in a CSV log. This project supports a college deep learning assignment focused on model design, training, evaluation, and experiment comparison.

## Dataset

**MNIST** — 60,000 training and 10,000 test grayscale digit images (28×28 pixels), loaded via `keras.datasets.mnist.load_data()`. Labels are digits 0–9.

## Steps

1. **Load dataset** — fetch MNIST and inspect shapes
2. **Normalize and flatten** — scale pixels to [0, 1] and reshape to 784-dimensional vectors
3. **Build baseline ANN** — Sequential model: 784 → 128 (ReLU) → Dropout(0.2) → 64 (ReLU) → 10 (softmax)
4. **Manual experiment tracking** — log learning rate, epochs, batch size, validation/test accuracy, and training time
5. **Run configurations** — three experiments (EXP-01, EXP-02, EXP-03) with different hyperparameters
6. **Save best model** — persist the highest test-accuracy run to `models/baseline_ann.keras`
7. **Results summary** — comparison table, best config, and bar chart from `experiment_log.csv`

## Project structure

```
baseline-model-tracking/
├── data/
├── notebooks/
│   └── 01_baseline_model.ipynb
├── models/
├── experiment_log.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup and run

From this directory:

```bash
cd "DL Experiment 2/baseline-model-tracking"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_baseline_model.ipynb
```

In Jupyter: **Kernel → Restart & Run All**.

After a successful run you should have:

- `experiment_log.csv` — experiment metrics (committed as results artifact)
- `models/baseline_ann.keras` — saved best model (ignored by git via `.gitignore`)

## Notes

- Trained weight files (`*.keras`, `*.h5`) are gitignored; `experiment_log.csv` is tracked.
- The notebook uses TensorFlow/Keras `.keras` format for saving the baseline model.
