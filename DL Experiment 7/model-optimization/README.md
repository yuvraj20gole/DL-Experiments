# Model Optimization — Quantization & Pruning on MNIST Baseline ANN

College deep learning assignment: apply post-training quantization (TFLite) and magnitude-based pruning (TFMOT) to the trained MNIST baseline ANN from Experiment 2, then compare size, accuracy, and latency.

## Objective

- Evaluate the Experiment 2 baseline ANN (size, accuracy, latency)
- Generate a quantized TFLite model
- Generate a 50%-sparsity pruned model
- Compare all three and select a deployment-ready version

## Dependency on Experiment 2

This project expects `models/baseline_ann.keras` (copied from  
`DL Experiment 2/baseline-model-tracking/models/baseline_ann.keras`).  
If that file is missing, re-run Experiment 2 first, then copy the saved model here.

## Project structure

```
model-optimization/
├── models/          # baseline_ann.keras (+ quantized / pruned outputs, gitignored)
├── notebooks/
│   └── 01_model_optimization.ipynb
├── results/         # comparison CSV + bar charts
├── requirements.txt
├── README.md
└── .gitignore
```

## How to run

Use the existing native **arm64** Python venv:

```bash
cd "DL Experiment 7/model-optimization"
source "../../DL Experiment 2/baseline-model-tracking/.venv/bin/activate"
pip install -r requirements.txt
jupyter notebook notebooks/01_model_optimization.ipynb
```

Or execute headlessly:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_model_optimization.ipynb \
  --ExecutePreprocessor.kernel_name=arm64-venv \
  --inplace
```

After a successful run you should see:

- `results/optimization_comparison.csv`
- `results/size_comparison.png`
- `results/latency_comparison.png`
- `results/accuracy_comparison.png`

## Note on TFMOT / Keras 3

TFMOT requires the legacy `tf_keras` API. Section 4 rebuilds the architecture in `tf_keras`, transfers the original trained weights via `get_weights()` / `set_weights()`, verifies accuracy matches the Keras 3 baseline, then applies pruning.
