# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_missing(df: pd.DataFrame):
    """
    Check for missing values in the DataFrame.
    """
    print("\n=== Missing Values ===")
    print(df.isnull().sum())


def correlation_check(df: pd.DataFrame, threshold: float = 0.7):
    """
    Check for highly correlated features (numeric columns only).
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.empty:
        print("No numeric columns to check correlation.")
        return

    corr_matrix = numeric_df.corr()
    print("\n=== Correlation Matrix ===")
    print(corr_matrix)

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # List highly correlated pairs
    correlated_pairs = [
        (col1, col2, corr_matrix.loc[col1, col2])
        for col1 in corr_matrix.columns
        for col2 in corr_matrix.columns
        if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > threshold
    ]
    if correlated_pairs:
        print("\nHighly correlated feature pairs (>|threshold|):")
        for pair in correlated_pairs:
            print(pair)
    else:
        print("\nNo highly correlated pairs found.")
