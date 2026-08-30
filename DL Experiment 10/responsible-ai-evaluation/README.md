# Responsible AI & Model Evaluation — Titanic Classification

College deep learning assignment: evaluating a machine learning classifier beyond aggregate accuracy using confusion matrices, classification reports, subgroup fairness & bias audits (by Sex), and global/local SHAP (SHapley Additive exPlanations) explainability.

## Objective

- **Beyond Aggregate Accuracy:** Evaluate model performance across classes with precision, recall, and F1-score.
- **Fairness & Bias Auditing:** Measure subgroup performance disparities across sensitive attributes (`Sex`) to identify demographic bias and disparate impact.
- **Model Explainability:** Utilize SHAP (`TreeExplainer`) for global feature importance rankings and instance-level decision attribution.
- **Reproducible Pipeline Reuse:** Reuse the cleaning, feature engineering, and stratified splitting pipeline from **Experiment 1**.

## Dataset

Reuses the Titanic dataset via `DL Experiment 1/titanic-pipeline/src/pipeline.py` (891 passenger records, stratified 70/15/15 train/val/test split).

## Project Structure

```
responsible-ai-evaluation/
├── notebooks/
│   └── 01_model_evaluation.ipynb
├── reports/
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── bias_report.csv
│   ├── bias_by_sex.png
│   ├── fairness_metrics_by_group.csv
│   └── shap_summary.png
├── models/
│   └── titanic_rf_classifier.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

Use the existing native **arm64** Python virtual environment:

```bash
cd "DL Experiment 10/responsible-ai-evaluation"
source "../../DL Experiment 2/baseline-model-tracking/.venv/bin/activate"
pip install -r requirements.txt
jupyter notebook notebooks/01_model_evaluation.ipynb
```

Or execute headlessly top-to-bottom:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_model_evaluation.ipynb \
  --ExecutePreprocessor.kernel_name=arm64-venv \
  --inplace
```

## Output Artifacts

- `reports/classification_report.csv` — Class-wise precision, recall, F1, and support.
- `reports/confusion_matrix.png` — Seaborn heatmap of test set predictions.
- `reports/bias_report.csv` — Accuracy breakdown by Sex group.
- `reports/bias_by_sex.png` — Bar chart visualizing subgroup accuracy disparities.
- `reports/fairness_metrics_by_group.csv` — Subgroup precision, recall, F1, and accuracy.
- `reports/shap_summary.png` — Global SHAP beeswarm summary plot for the "Survived" class.
- `models/titanic_rf_classifier.pkl` — Trained Random Forest classifier serialized via joblib.
