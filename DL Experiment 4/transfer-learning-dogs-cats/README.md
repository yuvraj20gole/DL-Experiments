# Transfer Learning: ResNet50 on Dogs vs Cats

## Objective

Compare **feature extraction** (frozen ResNet50 base) vs **fine-tuning** (unfreeze the last 30 layers) on the Kaggle Dogs vs Cats dataset. This is a college deep learning assignment focused on transfer learning, training-time tradeoffs, and evaluation.

## Dataset

[Kaggle Dogs vs Cats](https://www.kaggle.com/competitions/dogs-vs-cats) — binary classification (`cat` vs `dog`). Images are downloaded via the Kaggle API, then split 80/10/10 into `dataset/train`, `dataset/val`, and `dataset/test`.

You need a **Kaggle API token**:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings)
2. **API → Create New Token** — this downloads `kaggle.json`
3. Upload `kaggle.json` into the Colab working directory (or this project folder locally)

## Run in Google Colab (recommended)

ResNet50 training is expensive. Use Colab with a **GPU**:

**Runtime → Change runtime type → GPU**

Then:

1. Upload this notebook (and `kaggle.json`)
2. Run **Section 0** Kaggle download cells
3. Run all remaining sections top to bottom
4. Download `outputs.zip` from Section 9
5. Extract locally into `models/` and `results/`, then commit those artifacts

## Local run (possible, but slow)

```bash
cd "DL Experiment 4/transfer-learning-dogs-cats"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_transfer_learning.ipynb
```

On Apple Silicon, use a native **arm64** Python venv (the Experiment 2 venv works if TensorFlow already imports there). CPU-only training can take hours.

## Workflow after Colab

```
outputs.zip
├── models/resnet50_finetuned.keras
└── results/
    ├── benchmark_comparison.csv
    ├── accuracy_comparison.png
    └── time_comparison.png
```

Unzip into this project folder, then commit `results/` (model weight files are gitignored).

## Project structure

```
transfer-learning-dogs-cats/
├── dataset/train|val|test   # populated after Section 1
├── notebooks/01_transfer_learning.ipynb
├── models/
├── results/
├── requirements.txt
├── README.md
└── .gitignore
```
