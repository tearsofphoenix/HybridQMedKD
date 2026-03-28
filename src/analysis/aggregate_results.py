import os
import json
import pandas as pd
from src.utils.metrics import aggregate_fold_metrics


def load_fold_csv(exp_name, output_dir="outputs/tables"):
    path = os.path.join(output_dir, f"{exp_name}_folds.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).to_dict(orient="records")


def build_summary_from_csvs(exp_names, output_dir="outputs/tables"):
    summary = {}
    for name in exp_names:
        records = load_fold_csv(name, output_dir)
        if records:
            summary[name] = aggregate_fold_metrics(records)
    return summary


def save_summary(summary, output_dir="outputs/tables"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_dir}/summary.json")


if __name__ == "__main__":
    exp_names = [
        "student_classic",
        "student_classic_kd",
        "student_hybrid_nokd",
        "student_hybrid_kd",
    ]
    summary = build_summary_from_csvs(exp_names)
    save_summary(summary)
    for k, v in summary.items():
        print(f"\n{k}:")
        for metric, stats in v.items():
            if isinstance(stats, dict):
                print(f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
