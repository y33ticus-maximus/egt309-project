import json
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pandas.api.types as ptypes
import seaborn as sns
import yaml

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")



# 1. Load configuration
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 2. Load cleaned dataset
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
        print(f"[1] Loaded {len(df)} rows, {df.shape[1]} columns from {self.csv_path}.")
        return df



# 3. OneHotEncoder
def create_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)   # New version: sparse_output=False
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # Old version: sparse=False  
    
    #incase either version fails, return default OneHotEncoder
    



# 4. Prepare data
def prepare_data(cfg, df):
    target_col = cfg["data"]["target_column"]
    drop_columns = cfg["data"].get("drop_columns", []) or []

    df = df.copy()

    for col in drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_columns = [
        col for col in X.columns
        if ptypes.is_numeric_dtype(X[col])
    ]

    categorical_columns = [
        col for col in X.columns
        if not ptypes.is_numeric_dtype(X[col])
    ]

    print(f"[2] Numeric columns: {numeric_columns}")
    print(f"[2] Categorical columns: {categorical_columns}")

    stratify = y if cfg["split"].get("stratify", True) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=stratify
    )

    print(f"[2] Training rows: {len(X_train)}")
    print(f"[2] Testing rows : {len(X_test)}")

    print("\n[2] Class balance before SMOTE:")
    print(y_train.value_counts())

    return X_train, X_test, y_train, y_test, numeric_columns, categorical_columns


# ============================================================
# 5. Create SMOTE sampling strategy
# ============================================================
def create_partial_smote_strategy(y_train, ratio):
    class_counts = y_train.value_counts()
    max_count = class_counts.max()

    strategy = {}

    for class_name, count in class_counts.items():
        if count < max_count:
            new_count = int(count + (max_count - count) * ratio)
            strategy[class_name] = new_count

    return strategy


# ============================================================
# 6. Build pipeline
# ============================================================
def build_pipeline(
    cfg,
    numeric_columns,
    categorical_columns,
    smote_strategy,
    smote_k_neighbors,
    C_value
):
    scale_features = cfg["preprocessing"].get("scale_features", True)

    if scale_features:
        numeric_transformer = StandardScaler()
    else:
        numeric_transformer = "passthrough"

    categorical_transformer = create_onehot_encoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns)
        ],
        remainder="drop"
    )

    smote_random_state = cfg["resampling"].get("random_state", 42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(
                random_state=smote_random_state,
                sampling_strategy=smote_strategy,
                k_neighbors=smote_k_neighbors
            )),
            ("clf", LogisticRegression(
                C=C_value,
                solver="lbfgs",
                penalty="l2",
                max_iter=1000
            ))
        ]
    )

    return pipeline


# 7. training

def train_model_fast(cfg, X_train, y_train, numeric_columns, categorical_columns):
    print("\n[3] Fast training Logistic Regression with SMOTE...")

    class_counts = y_train.value_counts()
    min_class_count = class_counts.min()

    if min_class_count < 2:
        raise ValueError(
            "SMOTE cannot run because one class has fewer than 2 samples."
        )

    smote_k_neighbors = min(5, min_class_count - 1)

    # Small validation split for fast model selection
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=cfg["split"]["random_state"],
        stratify=y_train
    )

    # Only test a few combinations so it runs fast
    smote_ratios = [0.25, 0.40]
    C_values = [0.1, 1, 10]

    best_model = None
    best_score = -1
    best_params = None

    total_runs = len(smote_ratios) * len(C_values)
    run_count = 1

    for ratio in smote_ratios:
        smote_strategy = create_partial_smote_strategy(y_fit, ratio)

        for C_value in C_values:
            print(f"[3] Run {run_count}/{total_runs}: SMOTE ratio={ratio}, C={C_value}")

            model = build_pipeline(
                cfg,
                numeric_columns,
                categorical_columns,
                smote_strategy,
                smote_k_neighbors,
                C_value
            )

            model.fit(X_fit, y_fit)

            val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)

            print(f"    Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_score:
                best_score = val_accuracy
                best_model = model
                best_params = {
                    "smote_ratio": ratio,
                    "smote_strategy": smote_strategy,
                    "smote_k_neighbors": smote_k_neighbors,
                    "C": C_value,
                    "solver": "lbfgs",
                    "penalty": "l2"
                }

            run_count += 1

    print("\n[3] Best fast tuning parameters:")
    print(best_params)
    print(f"[3] Best validation accuracy: {best_score:.4f}")


    # After selecting the best settings, retrain on the full training set
    final_smote_strategy = create_partial_smote_strategy(
        y_train,
        best_params["smote_ratio"]
    )

    final_model = build_pipeline(
        cfg,
        numeric_columns,
        categorical_columns,
        final_smote_strategy,
        best_params["smote_k_neighbors"],
        best_params["C"]
    )

    final_model.fit(X_train, y_train)

    return final_model, best_params, best_score



# 8. Evaluate model
def evaluate_model(cfg, model, X_train, y_train, X_test, y_test, best_params, best_val_score):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Weighted average is used because the classes are imbalanced
    precision = precision_score(
        y_test,
        y_test_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_test_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        y_test_pred,
        average="weighted",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_test_pred,
        average="macro",
        zero_division=0
    )

    scores = {
        "model": "fast_logistic_regression_smote",
        "training_accuracy": train_accuracy,
        "testing_accuracy": test_accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "best_validation_accuracy": best_val_score,
        "best_params": best_params
    }

    print("\n=== Logistic Regression + SMOTE Results ===")
    print(f"Training Accuracy : {train_accuracy:.4f}")
    print(f"Testing Accuracy  : {test_accuracy:.4f}")
    print(f"Precision Weighted: {precision:.4f}")
    print(f"Recall Weighted   : {recall:.4f}")
    print(f"F1 Weighted       : {f1_weighted:.4f}")
    print(f"F1 Macro          : {f1_macro:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(
        y_test,
        y_test_pred,
        zero_division=0
    ))

    out_dir = cfg["evaluation"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if cfg["evaluation"].get("save_confusion_matrix", True):
        class_labels = list(model.named_steps["clf"].classes_)

        cm = confusion_matrix(
            y_test,
            y_test_pred,
            labels=class_labels
        )

        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Logistic Regression + SMOTE")
        plt.tight_layout()

        cm_path = os.path.join(out_dir, "confusion_matrix_logistic_regression_smote.png")
        plt.savefig(cm_path, dpi=120)
        plt.close()

        print(f"\n[4] Saved confusion matrix to: {cm_path}")

    metrics_path = os.path.join(out_dir, "metrics_summary_logistic_regression_smote.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"[4] Saved metrics to: {metrics_path}")

    return scores


# ============================================================
# 9. Save model
# ============================================================
def save_model(cfg, model):
    model_dir = cfg["output"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "logistic_regression_smote_pipeline.joblib")

    joblib.dump(model, model_path)

    print(f"\n[5] Saved full pipeline model to: {model_path}")
    print("[5] Done.")


# 10. Main

def main():
    start_time = time.time()

    cfg = load_config("config.yaml")

    loader = DataLoader(cfg["data"]["csv_path"])
    df = loader.load()

    X_train, X_test, y_train, y_test, numeric_columns, categorical_columns = prepare_data(cfg, df)

    model, best_params, best_val_score = train_model_fast(
        cfg,
        X_train,
        y_train,
        numeric_columns,
        categorical_columns
    )

    evaluate_model(
        cfg,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        best_params,
        best_val_score
    )

    save_model(cfg, model)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
