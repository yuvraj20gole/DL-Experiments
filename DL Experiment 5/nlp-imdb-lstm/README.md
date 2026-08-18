# IMDB Sentiment Classification with Embedding + LSTM

## Objective

Train a text classification model on IMDB movie reviews using an **Embedding** layer plus **LSTM**, then evaluate with precision, recall, F1-score, and a confusion matrix.

## Dataset

**IMDB Reviews** is a built-in Keras dataset (`keras.datasets.imdb`). No manual download is required. It contains 25,000 labeled train reviews and 25,000 test reviews (positive/negative). This notebook keeps the top 10,000 words and pads sequences to length 200.

## Project structure

```
nlp-imdb-lstm/
├── notebooks/01_nlp_pipeline.ipynb
├── models/
├── results/
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup and run

On this machine, reuse the existing native **arm64** venv (Experiment 2) so TensorFlow does not crash under Rosetta:

```bash
cd "DL Experiment 5/nlp-imdb-lstm"
source "../DL Experiment 2/baseline-model-tracking/.venv/bin/activate"
pip install -r requirements.txt
jupyter notebook notebooks/01_nlp_pipeline.ipynb
```

Select kernel **Python (arm64 venv)** if available, then **Kernel → Restart & Run All**.

To create a new arm64 venv instead:

```bash
arch -arm64 /opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_nlp_pipeline.ipynb
```

## Outputs (after a successful run)

| Artifact | Location |
| --- | --- |
| Trained model | `models/imdb_lstm.keras` |
| Classification report | `results/classification_report.csv` |
| Training curves | `results/training_curves.png` |
| Confusion matrix | `results/confusion_matrix.png` |

Model weight files are gitignored; **results CSV/PNG are tracked**.
