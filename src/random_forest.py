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

class RandomForestPipeline:
    def load_config(path="config.yaml"):                         # from config file
     with open(path, "r", encoding="utf-8") as f:                # allows code to take variable from that file
        return yaml.safe_load(f)\
        
    config = load_config("config.yaml")
    split_config = config["split"]
    rf_param = config["model"]["random_forest"]

    
    
    def __init__(self):
        self.csv_path = self.config["data"]["csv_path"]
        # self.csv_path = "data/cleaned_data.csv"              # define paths to access data / dave model
        self.model_path = self.config["output"]["random_forest"]["model_path"]

        self.df = None      # defining varaible
        self.new_df = None

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.train_pred = None
        self.y_pred = None
                                                                     # define model params
        self.rf_model = RandomForestClassifier(           
            n_estimators = self.rf_param["n_estimators"],            # to make predictions more stable
            max_depth = self.rf_param["max_depth"],                  # limit depth to reduce overfitting
            min_samples_split = self.rf_param["min_samples_split"],  # split node when it has at least 2 samples
            min_samples_leaf = self.rf_param["min_samples_leaf"],    # each final leaf to contain at least 1 sample
            max_features = self.rf_param["max_features"],            # Uses only some features at each split to reduce overfitting
            random_state = self.rf_param["random_state"],
            class_weight = self.rf_param["class_weight"],            # Balanced to cater to class imbalance in dataset
            n_jobs = self.rf_param["n_jobs"]
        )


    def load_data(self):                                  # read csv file
        self.df = pd.read_csv(self.csv_path)
        self.new_df = self.df.copy()

    def train_test_split_data(self):                            # split dataset into training/testing
        self.X = self.new_df.drop("Activity Level", axis=1)     # Remove Activity level from feature dataset
        self.X = self.X.drop("Session ID", axis=1)              # drop session ID as it contains no meaningful data
        self.X = pd.get_dummies(self.X, drop_first=True)        # OHE, cvt categorical class columns into numeric to model can read

        self.y = self.new_df["Activity Level"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(  # train test split using config file
            self.X,
            self.y,
            test_size=self.split_config["test_size"],
            random_state=self.split_config["random_state"],
            stratify=self.y
        )

    def train_model(self):                                    # train Model
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

    def save_model(self):                          # save model
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
