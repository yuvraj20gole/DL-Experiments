# CNN on CIFAR-10 with Real-World Constraints

## Objective

Train a convolutional neural network on CIFAR-10 while addressing **class imbalance** and **overfitting** using **data augmentation** and **class weighting**. Compare a baseline model (no augmentation, balanced sampling removed) against an augmented, class-weighted model.

## Dataset

**CIFAR-10** — 50,000 training and 10,000 test color images (32×32×3), 10 classes. Loaded via `keras.datasets.cifar10.load_data()`.

This notebook simulates imbalance by subsampling a few classes, then applies sklearn `compute_class_weight('balanced', ...)`.

## Project structure

```
cnn-real-world-constraints/
├── data/
├── notebooks/01_cnn_cifar10.ipynb
├── models/
├── results/
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup and run

Use the **native arm64** Python environment on Apple Silicon (Rosetta/x86_64 Python can crash TensorFlow on import).

From this directory:

```bash
cd "DL Experiment 3/cnn-real-world-constraints"

# Reuse the arm64 venv from Experiment 2 (recommended on this machine):
source "../DL Experiment 2/baseline-model-tracking/.venv/bin/activate"

# Or create a new arm64 venv:
# arch -arm64 /opt/homebrew/bin/python3.11 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook notebooks/01_cnn_cifar10.ipynb
```

In Jupyter, select kernel **Python (arm64 venv)** if available, then **Kernel → Restart & Run All**.

## Runtime note

CIFAR-10 CNN training is **compute-heavy**. The notebook runs:

- ~10 epochs baseline (no augmentation)
- 20 epochs with augmentation + class weights

On **CPU**, this can take **30+ minutes**. A **GPU** or **Google Colab** is recommended if training is too slow locally.

## Outputs (after a successful run)

| Artifact | Location |
| --- | --- |
| Trained model | `models/cnn_cifar10.keras` |
| Metrics table | `results/performance_comparison.csv` |
| Accuracy bar chart | `results/accuracy_comparison.png` |
| Overfitting curves | `results/overfitting_comparison.png` |

Model weight files are gitignored; **results CSV/PNG are tracked** as assignment artifacts.
