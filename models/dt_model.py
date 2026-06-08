from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib


def main():
    # Ensure new_df is available from previous cells
    if 'new_df' not in globals():
        try:
            df = pd.read_csv("cleaned_data.csv")
        except FileNotFoundError:
            print(
                "Error: 'new_df' (cleaned data) not found. Please ensure previous data cleaning steps were run or 'cleaned_data.csv' exists.")
            return None  # Return None if data is not found
    else:
        df = new_df.copy()  # Use new_df if available

    # Encode target variable
    le = LabelEncoder()
    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    df_processed['Activity Level Encoded'] = le.fit_transform(df_processed['Activity Level'])

    # Select features and target
    # Exclude non-numeric and ID-like columns, and the original target
    features = df_processed.drop(
        columns=['Time of Day', 'Session ID', 'HVAC Operation Mode', 'Ambient Light Level', 'Activity Level',
                 'Activity Level Encoded'])
    target = df_processed['Activity Level Encoded']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=16,
                                                        stratify=target)

    # Initialize Decision Tree Classifier
    dt_model = DecisionTreeClassifier(
        random_state=16,
        max_depth=None,
        max_leaf_nodes=None,
        max_features=None,
        min_samples_leaf=6,
        class_weight="balanced"
    )

    # Train model
    dt_model.fit(X_train, y_train)

    # Make predictions
    y_pred_dt = dt_model.predict(X_test)

    # Evaluate the model
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
    print("Decison Tree Macro F1-score:", f1_score(y_test, y_pred_dt, average="macro"))
    print("\nDecision Tree Classification Report:")
    print(classification_report(y_test, y_pred_dt, target_names=le.classes_))

    # Confusion Matrix
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=le.classes_)
    disp_dt.plot(cmap=plt.cm.Blues)
    plt.title("Decision Tree Confusion Matrix")
    plt.show()

    # Feature Importance Visualization
    if dt_model.feature_importances_ is not None and len(dt_model.feature_importances_) > 0:
        feature_importances = pd.Series(dt_model.feature_importances_, index=features.columns)
        sorted_feature_importances = feature_importances.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sorted_feature_importances.plot(kind='barh')
        plt.title('Decision Tree Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.gca().invert_yaxis()  # To have the most important feature at the top
        plt.tight_layout()
        plt.show()
    else:
        print("Feature importances could not be calculated or are all zero.")

    return dt_model  # Return the trained model


if __name__ == "__main__":
    # Call main and store the returned model
    dt_model_trained = main()
    if dt_model_trained is not None:
        # Save the trained model
        joblib.dump(dt_model_trained, 'decision_tree_model.joblib')
        print("Model saved as 'decision_tree_model.joblib'")
    else:
        print("Model training failed or data not found, so model was not saved.")