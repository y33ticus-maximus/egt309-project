# -*- coding: utf-8 -*-
"""
Logistic Regression Pipeline - ElderGuard Activity Level Prediction
===================================================================

End-to-end machine learning pipeline in ONE file:
    1. Ingest data from SQLite  (data/gas_monitoring.db)
    2. Clean + feature engineer (handle missing values, outliers, encoding)
    3. Train a Logistic Regression model
    4. Evaluate (accuracy, F1, confusion matrix)
    5. Save the trained model

All settings live in config.yaml so nothing here needs to be edited to retune.

Run it with:   python logistic_regression.py
"""

import json
import os
import sqlite3

import joblib
import matplotlib
matplotlib.use("Agg")  # lets plots save to file without a display (needed in Docker)
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


# =============================================================================
# 1. LOAD CONFIG
# =============================================================================
def load_config(path="config.yaml"):
    """Read all pipeline settings from the YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# 2. DATA INGESTION (SQLite)
# =============================================================================
class DataLoader:
    """Loads the monitoring dataset from a SQLite database into a DataFrame."""

    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name

    def load(self):
        """Connect to the database and read the configured table."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Database not found at '{self.db_path}'. "
                "Put gas_monitoring.db inside the data/ folder."
            )
        with sqlite3.connect(self.db_path) as conn:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )["name"].tolist()
            if self.table_name not in tables:
                raise ValueError(
                    f"Table '{self.table_name}' not found. Available: {tables}. "
                    "Update data.table_name in config.yaml."
                )
            df = pd.read_sql(f"SELECT * FROM {self.table_name}", conn)
        print(f"[1] Loaded {len(df)} rows, {df.shape[1]} columns from SQLite.")
        return df


# =============================================================================
# 3. PREPROCESSING (clean + feature engineering)
# =============================================================================
class DataPreprocessor:
    """Cleans the data and produces model-ready train/test splits.

    Anything that "learns" from the data (imputation values, scaler, resampler)
    is fit on the TRAINING split only, so no information leaks from the test set.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.target_col = cfg["data"]["target_column"]
        self.drop_columns = cfg["data"].get("drop_columns", []) or []
        self.label_encoder = LabelEncoder()
        self.scaler = None
        self.feature_columns = None
        self.numeric_columns = []
        self.categorical_columns = []

    def prepare(self, df):
        """Run the full cleaning flow. Returns X_train, X_test, y_train,
        y_test, class_names."""
        # --- basic cleaning -------------------------------------------------
        df = df.copy()
        for col in self.drop_columns:           # drop ID columns (e.g. Session ID)
            if col in df.columns:
                df = df.drop(columns=col)
        if self.cfg["preprocessing"].get("remove_duplicates", True):
            before = len(df)
            df = df.drop_duplicates()
            print(f"[2] Removed {before - len(df)} duplicate rows.")
        df = df.dropna(subset=[self.target_col])  # need a label to learn

        # --- identify column types -----------------------------------------
        features = [c for c in df.columns if c != self.target_col]
        self.categorical_columns = [c for c in features
                                    if not ptypes.is_numeric_dtype(df[c])]
        self.numeric_columns = [c for c in features
                                if c not in self.categorical_columns]
        print(f"[2] Numeric: {self.numeric_columns}")
        print(f"[2] Categorical: {self.categorical_columns}")

        # --- split first (so we only learn stats from training data) -------
        y = df[self.target_col]
        stratify = y if self.cfg["split"].get("stratify", True) else None
        train_df, test_df = train_test_split(
            df,
            test_size=self.cfg["split"]["test_size"],
            random_state=self.cfg["split"]["random_state"],
            stratify=stratify,
        )

        # --- clean + engineer (fit on train, apply to test) ----------------
        train_df = self._transform(train_df, fit=True)
        test_df = self._transform(test_df, fit=False)

        X_train = train_df[self.feature_columns]
        X_test = test_df[self.feature_columns]
        y_train = self.label_encoder.fit_transform(train_df[self.target_col])
        y_test = self.label_encoder.transform(test_df[self.target_col])

        # --- scale numeric features ----------------------------------------
        if self.cfg["preprocessing"].get("scale_features", True):
            self.scaler = StandardScaler()
            X_train = pd.DataFrame(self.scaler.fit_transform(X_train),
                                   columns=self.feature_columns, index=X_train.index)
            X_test = pd.DataFrame(self.scaler.transform(X_test),
                                  columns=self.feature_columns, index=X_test.index)

        # --- balance classes on the training set only ----------------------
        X_train, y_train = self._resample(X_train, y_train)

        return X_train, X_test, y_train, y_test, list(self.label_encoder.classes_)

    def _transform(self, df, fit):
        """Impute missing values, clip outliers, and one-hot encode."""
        df = df.copy()

        # impute (median for numbers, mode for categories)
        if fit:
            self._num_fill = {c: df[c].median() for c in self.numeric_columns}
            self._cat_fill = {c: (df[c].mode().iloc[0] if not df[c].mode().empty
                                  else "missing") for c in self.categorical_columns}
        for c in self.numeric_columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(self._num_fill[c])
        for c in self.categorical_columns:
            df[c] = df[c].fillna(self._cat_fill[c])

        # clip outliers using the IQR rule (bounds learned on train)
        oh = self.cfg["preprocessing"].get("outlier_handling", {})
        if oh.get("enabled"):
            mult = oh.get("iqr_multiplier", 1.5)
            if fit:
                self._bounds = {}
                for c in self.numeric_columns:
                    q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
                    iqr = q3 - q1
                    self._bounds[c] = (q1 - mult * iqr, q3 + mult * iqr)
            for c in self.numeric_columns:
                low, high = self._bounds[c]
                df[c] = df[c].clip(low, high)

        # one-hot encode categorical columns (this is what the model needs)
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


# =============================================================================
# 4. TRAIN  (Logistic Regression)
# =============================================================================
def train_model(cfg, X_train, y_train):
    """Build and fit a Logistic Regression model using config hyperparameters."""
    params = cfg["model"]["logistic_regression"]
    print(f"[3] Training Logistic Regression with params: {params}")
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


# =============================================================================
# 5. EVALUATE
# =============================================================================
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


# =============================================================================
# 6. MAIN  (runs the whole pipeline top to bottom)
# =============================================================================
def main():
    cfg = load_config("config.yaml")

    # 1. ingest
    loader = DataLoader(cfg["data"]["db_path"], cfg["data"]["table_name"])
    df = loader.load()

    # 2. preprocess
    pre = DataPreprocessor(cfg)
    X_train, X_test, y_train, y_test, class_names = pre.prepare(df)

    # 3. train
    model = train_model(cfg, X_train, y_train)

    # 4. evaluate
    evaluate_model(cfg, model, X_test, y_test, class_names)

    # 5. save model + preprocessor
    model_dir = cfg["output"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "logistic_regression.joblib"))
    joblib.dump(pre, os.path.join(model_dir, "preprocessor.joblib"))
    print(f"\n[5] Saved model to {model_dir}/. Done.")


if __name__ == "__main__":
    main()
