# Titanic Preprocessing Pipeline

## Objective

This project provides a reusable data preprocessing pipeline for the classic Titanic survival dataset. It is designed for a college assignment focused on cleaning, feature engineering, train/validation/test splitting, and sklearn-based transformation.

## Dataset

The pipeline uses the Titanic passenger dataset with features such as `Age`, `Fare`, `Sex`, `Embarked`, `Pclass`, and the target column `Survived`.

If `data/raw/titanic.csv` is not present locally, the pipeline will try to fetch it automatically:

1. Use `seaborn.load_dataset("titanic")` and save it to `data/raw/titanic.csv`
2. If that fails, download from the public fallback URL via `pandas.read_csv()`
3. If all automatic methods fail, manually download the dataset from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data) or run:

```python
import seaborn as sns
sns.load_dataset("titanic").to_csv("data/raw/titanic.csv", index=False)
```

Then place the file at `data/raw/titanic.csv`.

## Pipeline Steps

1. **Load data** — read CSV from `data/raw/titanic.csv` (or fetch it automatically)
2. **Clean** — drop `Cabin`, `Ticket`, `Name`, and `PassengerId`; fill missing `Embarked` with the mode
3. **Feature engineering** — add `FamilySize` (`SibSp + Parch + 1`) and `IsAlone`
4. **Split** — stratified 70/15/15 train/validation/test split with no leakage
5. **Transform** — fit preprocessing on train only, then transform validation and test sets using:
   - Numeric (`Age`, `Fare`, `FamilySize`): median imputation + `StandardScaler`
   - Categorical (`Sex`, `Embarked`, `Pclass`): most frequent imputation + `OneHotEncoder`

## Project Structure

```
titanic-pipeline/
├── data/raw/            # Raw dataset location (CSV ignored by git)
├── src/pipeline.py      # Preprocessing pipeline and split utilities
├── notebooks/explore.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Pipeline

From the project root:

```bash
python src/pipeline.py
```

This will fetch/load the dataset, preprocess it, split it, fit on the training set, transform validation and test sets, and print the resulting shapes.

## Explore in a Notebook

```bash
jupyter notebook notebooks/explore.ipynb
```

The notebook imports the pipeline, loads data, runs cleaning and feature engineering, and displays a quick sanity check with `head()` and missing-value counts.
