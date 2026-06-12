import json
import os
import sqlite3

import joblib
import matplotlib
matplotlib.use("Agg") #save plots as imges instead of showing them interactively
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

#read all configuration settings from config.yaml, which includes paths, model hyperparameters, and evaluation options
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


#load cleaned_data set

class DataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"File not found at '{self.csv_path}'. "
                "Run data_cleaning.py first to create cleaned_data.csv."
            )
        df = pd.read_csv(self.csv_path)
        print(f"[1] Loaded {len(df)} rows, {df.shape[1]} columns from cleaned_data.csv.")
        return df

#prepare data for modeling, including encoding, scaling, and resampling
class FeaturePreparer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.target_col = cfg["data"]["target_column"]     #column predicted:(Activity level)
        self.drop_columns = cfg["data"].get("drop_columns", []) or []   #columns to remove (SessionID)
        self.label_encoder = LabelEncoder()  #turns categorical labels into numeric values
        self.scaler = None
        self.feature_columns = None
        self.categorical_columns = []

    def prepare(self, df):
        df = df.copy()


        #drop ID columns = not useful for prediction and can introduce noise, so we remove them before modeling

        for col in self.drop_columns:
            if col in df.columns:
                df = df.drop(columns=col)

        #find which feature columns are text (need to be encoded into numbers)

        features = [c for c in df.columns if c != self.target_col]
        self.categorical_columns = [c for c in features
                                    if not ptypes.is_numeric_dtype(df[c])]
        print(f"[2] Categorical to encode: {self.categorical_columns}")

        # split into train and test FIRST, so the test set stays unseen (prevents data leakage)

        y = df[self.target_col]
        stratify = y if self.cfg["split"].get("stratify", True) else None  #keep class balance in both splits 
        train_df, test_df = train_test_split(
            df,
            test_size=self.cfg["split"]["test_size"],
            random_state=self.cfg["split"]["random_state"],
            stratify=stratify,
        )

        # one-hot encode the text columns into 0/1 columns
        train_df = self._encode(train_df, fit=True)
        test_df = self._encode(test_df, fit=False)

        X_train = train_df[self.feature_columns]
        X_test = test_df[self.feature_columns]
        y_train = self.label_encoder.fit_transform(train_df[self.target_col])
        y_test = self.label_encoder.transform(test_df[self.target_col])

        # scale the numbers to a similar range
        if self.cfg["preprocessing"].get("scale_features", True):
            self.scaler = StandardScaler()
            X_train = pd.DataFrame(self.scaler.fit_transform(X_train),
                                   columns=self.feature_columns, index=X_train.index)
            X_test = pd.DataFrame(self.scaler.transform(X_test),
                                  columns=self.feature_columns, index=X_test.index)
        #balanced the classes with SMOTE = applied to the training data
        X_train, y_train = self._resample(X_train, y_train)

        return X_train, X_test, y_train, y_test, list(self.label_encoder.classes_)

    def _encode(self, df, fit):
        df = pd.get_dummies(df, columns=self.categorical_columns, dummy_na=False)
        if fit:
            self.feature_columns = [c for c in df.columns if c != self.target_col]
        else:
            df = df.reindex(columns=self.feature_columns + [self.target_col],
                            fill_value=0)
        return df
    # Balance the classes using the method chosen in config (SMOTE)
    def _resample(self, X, y):
        method = (self.cfg["resampling"].get("method", "none") or "none").lower()
        rs = self.cfg["resampling"].get("random_state", 42)
        samplers = {"smote": SMOTE(random_state=rs),   #creates synthetic minority samples
                    "random_over": RandomOverSampler(random_state=rs),  #copies minority samples to balance classes
                    "random_under": RandomUnderSampler(random_state=rs)}  #removes majority samples to balance classes
        if method == "none":
            return X, y
        X_res, y_res = samplers[method].fit_resample(X, y)
        print(f"[2] Resampling '{method}': {len(y)} -> {len(y_res)} training rows.")
        return X_res, y_res

#train logistic regression model using the prepared training data and specified hyperparameters from config.yaml
def train_model(cfg, X_train, y_train):
    params = cfg["model"]["logistic_regression"]
    print(f"[3] Training Logistic Regression with params: {params}")
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(cfg, model, X_test, y_test, class_names):
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

    loader = DataLoader(cfg["data"]["csv_path"])
    df = loader.load()

    prep = FeaturePreparer(cfg)
    X_train, X_test, y_train, y_test, class_names = prep.prepare(df)

    model = train_model(cfg, X_train, y_train)
    evaluate_model(cfg, model, X_test, y_test, class_names)

    # 5. save the trained model and the preparer so they can be reused without retraining
    model_dir = cfg["output"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "logistic_regression.joblib"))

    # Save the fitted preparer (the encoder + scaler settings) so new data can be
    # prepared exactly the same way before predicting
    joblib.dump(prep, os.path.join(model_dir, "preprocessor.joblib"))
    print(f"\n[5] Saved model to {model_dir}/. Done.")


if __name__ == "__main__":
    main()

