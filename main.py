# main.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings

from src.load_data import load_csv
from src.eda import check_missing, correlation_check
from src.preprocessing.preprocessing import preprocess_data

warnings.filterwarnings("ignore", category=UserWarning)

# Load data
DATA_PATH = "data/Fraud.csv"
df = load_csv(DATA_PATH)

# EDA
print("\n=== Missing Values ===")
check_missing(df)
print("\n=== Correlation Matrix ===")
correlation_check(df)

# Preprocessing
target_column = "isFraud"
X, y, engineered_features = preprocess_data(df, target_column=target_column)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train XGBoost classifier
model = xgb.XGBClassifier(
    use_label_encoder=False, eval_metric="logloss", random_state=42
)
model.fit(X_train_res, y_train_res)

# Evaluate
y_pred = model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
print("\nModel and features saved successfully!")
