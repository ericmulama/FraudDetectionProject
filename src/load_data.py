import pandas as pd


def load_csv(filepath):
    """
    Load CSV file into a pandas DataFrame
    """
    df = pd.read_csv(filepath)
    return df
