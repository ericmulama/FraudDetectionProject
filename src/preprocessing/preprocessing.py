# preprocessing.py
import pandas as pd
import joblib


def preprocess_data(df, target_column="isFraud"):
    """
    Preprocess the fraud dataset:
    - Create smart features
    - One-hot encode categorical features
    - Return features (X), target (y), and list of feature names
    """
    # Drop rows with missing target
    df = df.dropna(subset=[target_column])

    # Feature engineering: ratios
    df["amount_to_oldbalance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1e-9)
    df["amount_to_newbalance_ratio"] = df["amount"] / (df["newbalanceOrig"] + 1e-9)
    df["balance_diff_org"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # One-hot encode 'type'
    df = pd.get_dummies(df, columns=["type"], prefix="type")

    # Features and target
    y = df[target_column]
    X = df.drop(columns=["nameOrig", "nameDest", target_column, "isFlaggedFraud"])

    # Save the list of features for dashboard
    engineered_features = X.columns.tolist()
    joblib.dump(engineered_features, "features.pkl")

    return X, y, engineered_features
