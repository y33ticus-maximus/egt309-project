from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib


if __name__ == "__main__":
    trainer = ActivityModelTrainer(
        data_path="data/cleaned_data.csv",
        model_type="decision_tree",
        model_path="saved_model/decision_tree_model.joblib"
    )

trainer.run_pipeline()