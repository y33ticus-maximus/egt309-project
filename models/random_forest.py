# ---------------- Import Libraries -----------------
import sqlite3
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

# Suppress noisy warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import yaml

def load_config(path="config.yaml"):                         # load data/variables from config file
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config("config.yaml")
    

class RandomForestPipeline:
    def __init__(self):
        self.csv_path = self.csv_path
        # self.csv_path = "data/cleaned_data.csv"              # define paths to access data / dave model
        self.model_path = "saved_model/rf_model.pkl"

        self.df = None
        self.new_df = None

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.rf_model = RandomForestClassifier(            # define model params
            n_estimators=1000,
            max_depth=70,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        self.train_pred = None
        self.y_pred = None

    def load_data(self):                                  # read csv file
        self.df = pd.read_csv(self.csv_path)
        self.new_df = self.df.copy()

    def train_test_split_data(self):                            # split dataset into training/testing
        self.X = self.new_df.drop("Activity Level", axis=1)
        self.X = self.X.drop("Session ID", axis=1)
        self.X = pd.get_dummies(self.X, drop_first=True)

        self.y = self.new_df["Activity Level"]

        split_config = config["split"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(  # train test split using config file
            self.X,
            self.y,
            test_size=split_config["test_size"],
            random_state=split_config["random_state"],
            stratify=self.y
        )

    def train_model(self):
        self.rf_model.fit(self.X_train, self.y_train)

        self.train_pred = self.rf_model.predict(self.X_train)
        self.y_pred = self.rf_model.predict(self.X_test)

    def evaluate_model(self):                                                                # model evaluation
        print("Training Accuracy:", accuracy_score(self.y_train, self.train_pred))           # compare acc/recall/precision/f1
        print("Testing Accuracy:", accuracy_score(self.y_test, self.y_pred))                 # f1 score is priority

        print("\nAccuracy:", accuracy_score(self.y_test, self.y_pred))
        print("Precision:", precision_score(self.y_test, self.y_pred, average="weighted"))
        print("Recall:", recall_score(self.y_test, self.y_pred, average="weighted"))
        print("F1 Score:", f1_score(self.y_test, self.y_pred, average="weighted"))

        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))

    def save_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.rf_model.classes_)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=self.rf_model.classes_, yticklabels=self.rf_model.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Random Forest")
        plt.tight_layout()
        plt.savefig("outputs/rf_confusion_matrix.png", dpi=120)
        plt.close()
        print("Saved outputs/rf_confusion_matrix.png")

    def save_feature_importance(self):                                                        # determine feature importance for later analysis
        feature_imp = pd.Series(self.rf_model.feature_importances_,
                                index=self.X.columns).sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        feature_imp.plot(kind="bar", color="seagreen")
        plt.ylabel("Relative Importance")
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        plt.savefig("outputs/rf_feature_importance.png", dpi=120)
        plt.close()
        print("Saved outputs/rf_feature_importance.png")

    def save_model(self):
        with open(self.model_path, "wb") as file:
            pickle.dump(self.rf_model, file)
        print("Saved saved_model/rf_model.pkl")

    def run(self):                         # pipeline
        self.load_data()
        self.train_test_split_data()
        self.train_model()
        self.evaluate_model()
        self.save_confusion_matrix()
        self.save_feature_importance()
        self.save_model()


if __name__ == "__main__":
    pipeline = RandomForestPipeline()
    pipeline.run()
