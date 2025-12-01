# src/features.py
import pandas as pd


def create_features(df):
    # Transaction ratios
    df["amount_oldbalance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["amount_newbalance_ratio"] = df["amount"] / (df["newbalanceOrig"] + 1)

    # Balance differences
    df["diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["diff_dest"] = df["oldbalanceDest"] - df["newbalanceDest"]

    # One-hot encode 'type'
    df = pd.get_dummies(df, columns=["type"], prefix="type")

    return df
