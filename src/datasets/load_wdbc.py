import pandas as pd
import numpy as np


def load_wdbc(csv_path, target_col="Diagnosis", id_col="ID"):
    """
    Load Breast Cancer Wisconsin (Diagnostic) dataset.
    Returns:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,), 1=Malignant, 0=Benign
        df: raw DataFrame
    """
    df = pd.read_csv(csv_path)
    y = df[target_col].map({"M": 1, "B": 0}).values.astype(np.float32)
    drop_cols = [target_col]
    if id_col in df.columns:
        drop_cols.append(id_col)
    X = df.drop(columns=drop_cols).values.astype(np.float32)
    return X, y, df
