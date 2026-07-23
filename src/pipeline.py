"""Reusable preprocessing pipeline for the Titanic dataset."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["Age", "Fare", "FamilySize"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Pclass"]
FALLBACK_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
COLUMN_ALIASES = {
    "survived": "Survived",
    "pclass": "Pclass",
    "name": "Name",
    "sex": "Sex",
    "age": "Age",
    "sibsp": "SibSp",
    "parch": "Parch",
    "ticket": "Ticket",
    "fare": "Fare",
    "cabin": "Cabin",
    "embarked": "Embarked",
    "passengerid": "PassengerId",
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to the Kaggle Titanic schema."""
    renamed = {
        column: COLUMN_ALIASES[column.lower()]
        for column in df.columns
        if column.lower() in COLUMN_ALIASES
    }
    return df.rename(columns=renamed)


def get_titanic_data(path: str = "data/raw/titanic.csv") -> pd.DataFrame:
    """Load Titanic data from disk or fetch it from an available source."""
    data_path = Path(path)
    if data_path.exists():
        return _standardize_columns(pd.read_csv(data_path))

    data_path.parent.mkdir(parents=True, exist_ok=True)
    errors = []

    try:
        import seaborn as sns

        df = _standardize_columns(sns.load_dataset("titanic"))
        df.to_csv(data_path, index=False)
        return df
    except Exception as exc:
        errors.append(f"seaborn.load_dataset failed: {exc}")

    try:
        df = _standardize_columns(pd.read_csv(FALLBACK_URL))
        df.to_csv(data_path, index=False)
        return df
    except Exception as exc:
        errors.append(f"URL download failed: {exc}")

    error_details = "\n".join(f"- {message}" for message in errors)
    raise RuntimeError(
        "Could not obtain Titanic dataset automatically.\n"
        f"{error_details}\n"
        f"Please manually place titanic.csv at: {data_path.resolve()}"
    )


def split_data(df: pd.DataFrame, target_col: str = "Survived"):
    """Split data into stratified 70/15/15 train/val/test sets."""
    features = df.drop(columns=[target_col])
    target = df[target_col]

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.30,
        stratify=target,
        random_state=42,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    train_df = x_train.copy()
    train_df[target_col] = y_train

    val_df = x_val.copy()
    val_df[target_col] = y_val

    test_df = x_test.copy()
    test_df[target_col] = y_test

    return train_df, val_df, test_df


class TitanicPreprocessor:
    """End-to-end cleaning, feature engineering, and sklearn transformation."""

    def __init__(self):
        self.numeric_features = NUMERIC_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        self.transformer = None

    def load_data(self, path: str) -> pd.DataFrame:
        """Load titanic.csv into a pandas DataFrame."""
        return _standardize_columns(pd.read_csv(path))

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unused columns and fill missing Embarked values."""
        cleaned = df.drop(
            columns=["Cabin", "Ticket", "Name", "PassengerId"],
            errors="ignore",
        ).copy()
        cleaned["Embarked"] = cleaned["Embarked"].fillna(cleaned["Embarked"].mode()[0])
        return cleaned

    def feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add FamilySize and IsAlone features."""
        engineered = df.copy()
        engineered["FamilySize"] = engineered["SibSp"] + engineered["Parch"] + 1
        engineered["IsAlone"] = (engineered["FamilySize"] == 1).astype(int)
        return engineered

    def build_transformer(self) -> ColumnTransformer:
        """Build sklearn ColumnTransformer for numeric and categorical features."""
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ]
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the transformer on df and return transformed features."""
        self.transformer = self.build_transformer()
        feature_frame = df[self.numeric_features + self.categorical_features]
        transformed = self.transformer.fit_transform(feature_frame)
        return pd.DataFrame(transformed)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform df using a fitted transformer."""
        if self.transformer is None:
            raise RuntimeError("Transformer is not fitted. Call fit_transform first.")

        feature_frame = df[self.numeric_features + self.categorical_features]
        transformed = self.transformer.transform(feature_frame)
        return pd.DataFrame(transformed)


if __name__ == "__main__":
    df = get_titanic_data()

    preprocessor = TitanicPreprocessor()
    df = preprocessor.clean(df)
    df = preprocessor.feature_engineer(df)

    train_df, val_df, test_df = split_data(df)

    x_train = preprocessor.fit_transform(train_df)
    x_val = preprocessor.transform(val_df)
    x_test = preprocessor.transform(test_df)

    print("Train shape:", x_train.shape)
    print("Validation shape:", x_val.shape)
    print("Test shape:", x_test.shape)
