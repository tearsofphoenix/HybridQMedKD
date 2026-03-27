import numpy as np


def aggregate_fold_metrics(all_metrics):
    """
    Compute mean and std across folds for each metric.
    """
    keys = [k for k in all_metrics[0].keys() if k not in ["fold"]]
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals))
        }
    return summary
