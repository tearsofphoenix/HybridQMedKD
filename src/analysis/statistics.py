import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_METRICS = ("auc", "f1", "acc", "mcc", "train_time", "infer_time")


def load_fold_records(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing fold CSV: {path}")
    return pd.read_csv(path).to_dict(orient="records")


def confidence_interval(values, confidence=0.95):
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if n < 2:
        return {
            "n": int(n),
            "mean": mean,
            "std": std,
            "ci_low": mean,
            "ci_high": mean,
        }
    sem = stats.sem(arr)
    t_crit = stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1)
    margin = float(t_crit * sem)
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
    }


def summarize_records(records, metrics=DEFAULT_METRICS, confidence=0.95):
    summary = {}
    if not records:
        return summary
    for metric in metrics:
        if metric not in records[0]:
            continue
        values = [row[metric] for row in records]
        summary[metric] = confidence_interval(values, confidence=confidence)
    return summary


def paired_statistics(reference, candidate, confidence=0.95):
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.shape != cand.shape:
        raise ValueError("Paired statistics require arrays with identical shape")
    diff = cand - ref
    base = confidence_interval(diff, confidence=confidence)
    if diff.size < 2:
        p_value = None
        statistic = None
    else:
        test = stats.ttest_rel(cand, ref)
        statistic = float(test.statistic) if np.isfinite(test.statistic) else None
        p_value = float(test.pvalue) if np.isfinite(test.pvalue) else None
    std_diff = float(np.std(diff, ddof=0))
    if std_diff == 0.0:
        effect_size_dz = 0.0 if float(np.mean(diff)) == 0.0 else None
    else:
        effect_size_dz = float(np.mean(diff) / std_diff)
    base.update(
        {
            "mean_diff": float(np.mean(diff)),
            "p_value": p_value,
            "t_statistic": statistic,
            "effect_size_dz": effect_size_dz,
        }
    )
    return base


def compare_record_sets(reference_records, candidate_records, metrics=("auc", "f1", "mcc", "acc"), confidence=0.95):
    comparisons = {}
    for metric in metrics:
        if metric not in reference_records[0] or metric not in candidate_records[0]:
            continue
        ref = [row[metric] for row in reference_records]
        cand = [row[metric] for row in candidate_records]
        comparisons[metric] = paired_statistics(ref, cand, confidence=confidence)
    return comparisons


def save_json(data, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return str(path)
