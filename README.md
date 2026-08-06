# DL Experiments

A collection of deep learning and machine learning experiments for coursework and practice.

## Experiments

| # | Folder | Description |
| --- | --- | --- |
| 1 | [DL Experiment 1/titanic-pipeline](DL%20Experiment%201/titanic-pipeline/) | Titanic data preprocessing pipeline (cleaning, feature engineering, sklearn transforms) |
| 2 | [DL Experiment 2/baseline-model-tracking](DL%20Experiment%202/baseline-model-tracking/) | MNIST baseline ANN with manual experiment tracking |
| 3 | [DL Experiment 3/cnn-real-world-constraints](DL%20Experiment%203/cnn-real-world-constraints/) | CIFAR-10 CNN with augmentation and class imbalance handling |

## Repository structure

```
DL-Experiments/
├── README.md
├── DL Experiment 1/
│   └── titanic-pipeline/
│       ├── README.md
│       ├── data/raw/
│       ├── src/
│       ├── notebooks/
│       └── requirements.txt
├── DL Experiment 2/
│   └── baseline-model-tracking/
│       ├── README.md
│       ├── experiment_log.csv
│       ├── notebooks/
│       └── requirements.txt
├── DL Experiment 3/
│   └── cnn-real-world-constraints/
│       ├── README.md
│       ├── results/
│       ├── notebooks/
│       └── requirements.txt
└── ...
```

Each experiment lives in its own numbered folder with a dedicated README and self-contained code.
