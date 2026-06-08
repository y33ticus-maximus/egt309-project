

import json
import os
import sqlite3

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to file (needed in Docker)
import matplotlib.pyplot as plt
import pandas as pd
import pandas.api.types as ptypes
import seaborn as sns
import yaml
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.filterwarnings("ignore")


def load_config(path="config.yaml"):
    """Read all pipeline settings from the YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



#LOAD the already-cleaned data from SQLite

class DataLoader:
    """Loads the cleaned dataset from a SQLite table into a DataFrame."""

    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name

    def load(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Database not found at '{self.db_path}'. "
                "Run data_cleaning.py first to create the cleaned_data table."
            )
        with sqlite3.connect(self.db_path) as conn:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )["name"].tolist()
            if self.table_name not in tables:
                raise ValueError(
                    f"Table '{self.table_name}' not found. Available: {tables}. "
                    "Run data_cleaning.py first, or update data.table_name in config.yaml."
                )
            df = pd.read_sql(f"SELECT * FROM {self.table_name}", conn)
        print(f"[1] Loaded {len(df)} rows, {df.shape[1]} columns from SQLite.")
        return df


# =============================================================================
# 2. MODEL-SPECIFIC FEATURE PREPARATION (no data cleaning here)
# =============================================================================
class FeaturePreparer:
    """Turns the cleaned DataFrame into model-ready train/test arrays.

    Only model preparation: drop identifier, split, one-hot encode, scale, and
    SMOTE. The scaler and SMOTE are fit on the TRAINING split only, so no
    information leaks from the test set.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.target_col = cfg["data"]["target_column"]
        self.drop_columns = cfg["data"].get("drop_columns", []) or []
        self.label_encoder = LabelEncoder()
        self.scaler = None
        self.feature_columns = None
        self.categorical_columns = []

    def prepare(self, df):
        """Return X_train, X_test, y_train, y_test, class_names."""
        df = df.copy()

        
        for col in self.drop_columns:
            if col in df.columns:
                df = df.drop(columns=col)


        features = [c for c in df.columns if c != self.target_col]
        self.categorical_columns = [c for c in features
                                    if not ptypes.is_numeric_dtype(df[c])]
        print(f"[2] Categorical to encode: {self.categorical_columns}")

    
        y = df[self.target_col]
        stratify = y if self.cfg["split"].get("stratify", True) else None
        train_df, test_df = train_test_split(
            df,
            test_size=self.cfg["split"]["test_size"],
            random_state=self.cfg["split"]["random_state"],
            stratify=stratify,
        )


        train_df = self._encode(train_df, fit=True)
        test_df = self._encode(test_df, fit=False)

        X_train = train_df[self.feature_columns]
        X_test = test_df[self.feature_columns]
        y_train = self.label_encoder.fit_transform(train_df[self.target_col])
        y_test = self.label_encoder.transform(test_df[self.target_col])

        if self.cfg["preprocessing"].get("scale_features", True):
            self.scaler = StandardScaler()
            X_train = pd.DataFrame(self.scaler.fit_transform(X_train),
                                   columns=self.feature_columns, index=X_train.index)
            X_test = pd.DataFrame(self.scaler.transform(X_test),
                                  columns=self.feature_columns, index=X_test.index)

        # balance classes on the training set only
        X_train, y_train = self._resample(X_train, y_train)

        return X_train, X_test, y_train, y_test, list(self.label_encoder.classes_)

    def _encode(self, df, fit):
        """One-hot encode categorical columns; align train/test feature sets."""
        df = pd.get_dummies(df, columns=self.categorical_columns, dummy_na=False)
        if fit:
            self.feature_columns = [c for c in df.columns if c != self.target_col]
        else:
            df = df.reindex(columns=self.feature_columns + [self.target_col],
                            fill_value=0)
        return df

    def _resample(self, X, y):
        """Balance the classes in the training data (config-driven)."""
        method = (self.cfg["resampling"].get("method", "none") or "none").lower()
        rs = self.cfg["resampling"].get("random_state", 42)
        samplers = {"smote": SMOTE(random_state=rs),
                    "random_over": RandomOverSampler(random_state=rs),
                    "random_under": RandomUnderSampler(random_state=rs)}
        if method == "none":
            return X, y
        X_res, y_res = samplers[method].fit_resample(X, y)
        print(f"[2] Resampling '{method}': {len(y)} -> {len(y_res)} training rows.")
        return X_res, y_res



def train_model(cfg, X_train, y_train):
    """Build and fit a Logistic Regression model using config hyperparameters."""
    params = cfg["model"]["logistic_regression"]
    print(f"[3] Training Logistic Regression with params: {params}")
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model



def evaluate_model(cfg, model, X_test, y_test, class_names):
    """Score the model and save a confusion-matrix figure."""
    y_pred = model.predict(X_test)
    scores = {
        "model": "logistic_regression",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }
    print("\n=== Logistic Regression results ===")
    print(f"Accuracy     : {scores['accuracy']:.4f}")
    print(f"F1 (weighted): {scores['f1_weighted']:.4f}")
    print(f"F1 (macro)   : {scores['f1_macro']:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=class_names, zero_division=0))

    out_dir = cfg["evaluation"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    if cfg["evaluation"].get("save_confusion_matrix", True):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Logistic Regression")
        plt.tight_layout()
        path = os.path.join(out_dir, "confusion_matrix.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"[4] Saved {path}")

    with open(os.path.join(out_dir, "metrics_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    return scores



def main():
    cfg = load_config("config.yaml")

    loader = DataLoader(cfg["data"]["db_path"], cfg["data"]["table_name"])
    df = loader.load()

    prep = FeaturePreparer(cfg)
    X_train, X_test, y_train, y_test, class_names = prep.prepare(df)

    model = train_model(cfg, X_train, y_train)
    evaluate_model(cfg, model, X_test, y_test, class_names)

    model_dir = cfg["output"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "logistic_regression.joblib"))
    joblib.dump(prep, os.path.join(model_dir, "preprocessor.joblib"))
    print(f"\n[5] Saved model to {model_dir}/. Done.")


if __name__ == "__main__":
    main()
