import pandas as pd
import numpy as np

from .load_wdbc import load_wdbc


def _encode_binary_target(series, positive_label=None, negative_label=None):
    if positive_label is not None and negative_label is not None:
        return series.map({positive_label: 1, negative_label: 0}).values.astype(np.float32)

    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.isna().any():
        unique_vals = sorted(set(numeric.tolist()))
        if unique_vals == [0, 1] or unique_vals == [0.0, 1.0]:
            return numeric.values.astype(np.float32)

    raise ValueError(
        "Binary target encoding failed. Please provide positive_label and negative_label "
        "for non-numeric binary targets."
    )


def load_binary_tabular_csv(
    csv_path,
    target_col,
    id_col=None,
    positive_label=None,
    negative_label=None,
    drop_cols=None,
):
    """
    Load a generic binary tabular CSV dataset.

    Returns:
        X: np.ndarray, shape (n_samples, n_features)
        y: np.ndarray, shape (n_samples,), binary labels in {0,1}
        df: raw DataFrame
    """
    df = pd.read_csv(csv_path)
    y = _encode_binary_target(
        df[target_col],
        positive_label=positive_label,
        negative_label=negative_label,
    )

    cols_to_drop = [target_col]
    if id_col and id_col in df.columns:
        cols_to_drop.append(id_col)
    if drop_cols:
        cols_to_drop.extend([c for c in drop_cols if c in df.columns])

    X = df.drop(columns=cols_to_drop).values.astype(np.float32)
    return X, y, df


def load_dataset(
    csv_path,
    dataset_name="wdbc",
    target_col=None,
    id_col=None,
    positive_label=None,
    negative_label=None,
    drop_cols=None,
):
    dataset_name = (dataset_name or "wdbc").lower()

    if dataset_name == "wdbc":
        return load_wdbc(
            csv_path,
            target_col=target_col or "Diagnosis",
            id_col=id_col or "ID",
        )

    return load_binary_tabular_csv(
        csv_path,
        target_col=target_col or "target",
        id_col=id_col,
        positive_label=positive_label,
        negative_label=negative_label,
        drop_cols=drop_cols,
    )
