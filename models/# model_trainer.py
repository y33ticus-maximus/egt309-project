# model_trainer.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class ActivityModelTrainer:
    def __init__(self, data_path, model_type="random_forest", model_path="saved_model/model.joblib"):
        self.data_path = data_path
        self.model_type = model_type
        self.model_path = model_path

        self.df = None
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.label_encoder = LabelEncoder()
        self.model = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print("Data loaded successfully.")
        print("Dataset shape:", self.df.shape)

    def prepare_data(self):
        df_processed = self.df.copy()

        # Encode target column
        df_processed["Activity Level Encoded"] = self.label_encoder.fit_transform(
            df_processed["Activity Level"]
        )

        # Separate features and target
        self.X = df_processed.drop(
            columns=[
                "Activity Level",
                "Activity Level Encoded"
            ]
        )

        self.y = df_processed["Activity Level Encoded"]

        # Drop Session ID because it is only an identifier
        if "Session ID" in self.X.columns:
            self.X = self.X.drop("Session ID", axis=1)

        # Use a copy for model-specific column removal so original processed data is preserved
        X_processed = self.X.copy()

        if self.model_type == "decision_tree":
            irrelevant_columns = [
                "Time of Day",
                "HVAC Operation Mode",
                "Ambient Light Level"
            ]
            X_processed = X_processed.drop(columns=[col for col in irrelevant_columns if col in X_processed.columns], errors="ignore")

        # One-hot encode categorical columns
        self.X = pd.get_dummies(X_processed)

        print("Data prepared successfully.")
        print("Feature shape:", self.X.shape)

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

        print("Train-test split completed.")
        print("Training rows:", self.X_train.shape[0])
        print("Testing rows:", self.X_test.shape[0])

    def choose_model(self):
        if self.model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )

        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=700,
                max_depth=50,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )

        elif self.model_type == "decision_tree":
            self.model = DecisionTreeClassifier(
                random_state=42,
                max_depth=None,
                max_leaf_nodes=None,
                max_features=None,
                min_samples_leaf=6,
                class_weight="balanced"
            )

        else:
            raise ValueError("Invalid model_type. Choose 'logistic_regression', 'random_forest', or 'decision_tree'.")

        print(f"{self.model_type} model selected.")

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def evaluate_model(self):
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)

        print("\nTraining Accuracy:", accuracy_score(self.y_train, train_pred))
        print("Testing Accuracy:", accuracy_score(self.y_test, test_pred))

        print("\nTesting Macro F1-score:", f1_score(self.y_test, test_pred, average="macro"))
        print("Testing Weighted F1-score:", f1_score(self.y_test, test_pred, average="weighted"))

        print("\nClassification Report:")
        print(
            classification_report(
                self.y_test,
                test_pred,
                target_names=self.label_encoder.classes_
            )
        )

    def plot_confusion_matrix(self):
        test_pred = self.model.predict(self.X_test)

        cm = confusion_matrix(self.y_test, test_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.label_encoder.classes_
        )

        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{self.model_type} Confusion Matrix")
        plt.show()

    def plot_feature_importance(self, top_n=15):
        if hasattr(self.model, "feature_importances_"):
            feature_importances = pd.Series(
                self.model.feature_importances_,
                index=self.X.columns
            ).sort_values(ascending=False)

            display_n = min(top_n, len(feature_importances))
            top_features = feature_importances.head(display_n)

            plt.figure(figsize=(10, 6))
            top_features.plot(kind="barh")
            plt.title(f"Top {display_n} Feature Importances - {self.model_type}")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        else:
            print(f"{self.model_type} does not provide feature_importances_.")

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def run_pipeline(self):
        self.load_data()
        self.prepare_data()
        self.split_data()
        self.choose_model()
        self.train_model()
        self.evaluate_model()
        self.plot_confusion_matrix()
        self.plot_feature_importance()
        self.save_model()


if __name__ == "__main__":
    trainer = ActivityModelTrainer(
        data_path="data/cleaned_data.csv",
        model_type="random_forest",
        model_path="saved_model/random_forest_model.joblib"
    )

# random forest as default as it's best performing, change model_type("logistic_regression","decision_tree") to use other models
    trainer.run_pipeline()