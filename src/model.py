# src/model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # Silence XGBoost warnings


def train_evaluate_model(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Oversample using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train XGBoost
    model = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # Feature list
    features = X.columns.tolist()

    # Layman-friendly summary
    fraud_detected = sum((y_pred == 1) & (y_test == 1))
    total_frauds = sum(y_test == 1)
    print(
        f"\nFraud detection accuracy: {fraud_detected}/{total_frauds} actual frauds detected."
    )

    return model, X_train, X_test, y_train, y_test, features
