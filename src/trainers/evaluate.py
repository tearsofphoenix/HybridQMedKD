import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score
)


def evaluate_binary(y_true, y_prob, threshold=0.5):
    """
    Evaluate binary classification predictions.

    Args:
        y_true: array-like of ground truth labels (0/1)
        y_prob: array-like of predicted probabilities
        threshold: decision threshold

    Returns:
        dict with auc, f1, acc, mcc, precision, recall
    """
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "acc": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
