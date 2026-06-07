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


# ---------------- Load Data (from SQLite) -----------------
# Task 2 requires the data to be fetched via SQLite from data/gas_monitoring.db
conn = sqlite3.connect("data/gas_monitoring.db")
df = pd.read_sql_query("SELECT * FROM gas_monitoring", conn)
conn.close()
new_df = df.copy()

# ---------------- Train Test Split -----------------
X = new_df.drop("Activity Level", axis=1)
X = X.drop("Session ID", axis=1)          # Session ID is an identifier, not a feature
X = pd.get_dummies(X, drop_first=True)    # one-hot encode the categorical columns

y = new_df["Activity Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- Random Forest Classifier -----------------
# max_depth and min_samples_leaf reduced from (50, 1) to (15, 5) to control
# overfitting (the deeper forest scored 100% on training but only ~67% on test).
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    class_weight="balanced",   # handles the class imbalance
    n_jobs=-1
)

# Train model
rf_model.fit(X_train, y_train)

train_pred = rf_model.predict(X_train)
y_pred = rf_model.predict(X_test)

# ---------------- Evaluation -----------------
print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix (saved to outputs/)
cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("outputs/rf_confusion_matrix.png", dpi=120)
plt.close()
print("Saved outputs/rf_confusion_matrix.png")

# ---------------- Feature Importance -----------------
# The problem statement asks which features matter most - Random Forest gives this.
feature_imp = pd.Series(rf_model.feature_importances_,
                        index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
feature_imp.plot(kind="bar", color="seagreen")
plt.ylabel("Relative Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("outputs/rf_feature_importance.png", dpi=120)
plt.close()
print("Saved outputs/rf_feature_importance.png")

# ---------------- Save Model -----------------
with open("saved_model/rf_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)
print("Saved saved_model/rf_model.pkl")
