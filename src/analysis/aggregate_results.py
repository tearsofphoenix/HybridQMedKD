import os
import json
import pandas as pd
from src.utils.metrics import aggregate_fold_metrics
from src.utils.io import get_tables_dir


def load_fold_csv(exp_name, output_dir=None):
    output_dir = output_dir or get_tables_dir()
    path = os.path.join(output_dir, f"{exp_name}_folds.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).to_dict(orient="records")


def build_summary_from_csvs(exp_names, output_dir=None):
    output_dir = output_dir or get_tables_dir()
    summary = {}
    for name in exp_names:
        records = load_fold_csv(name, output_dir)
        if records:
            summary[name] = aggregate_fold_metrics(records)
    return summary


def build_named_summary_map(exp_name_map, output_dir=None):
    output_dir = output_dir or get_tables_dir()
    summary = {}
    for key, exp_name in exp_name_map.items():
        records = load_fold_csv(exp_name, output_dir)
        if records:
            summary[key] = aggregate_fold_metrics(records)
    return summary


def build_ablation_position_summary(output_dir=None):
    return build_named_summary_map(
        {
            "front": "ablation_pos_front",
            "middle": "ablation_pos_middle",
            "tail": "ablation_pos_tail",
        },
        output_dir=output_dir,
    )


def build_ablation_pca_summary(output_dir=None):
    return build_named_summary_map(
        {
            4: "ablation_pca_4",
            6: "ablation_pca_6",
            8: "ablation_pca_8",
        },
        output_dir=output_dir,
    )


def save_json(data, filename, output_dir=None):
    output_dir = output_dir or get_tables_dir()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")
    return path


def save_available_ablation_summaries(output_dir=None):
    output_dir = output_dir or get_tables_dir()
    pos_summary = build_ablation_position_summary(output_dir)
    pca_summary = build_ablation_pca_summary(output_dir)

    if pos_summary:
        save_json(pos_summary, "ablation_position.json", output_dir)
    if pca_summary:
        save_json(pca_summary, "ablation_pca.json", output_dir)


def save_summary(summary, output_dir=None):
    output_dir = output_dir or get_tables_dir()
    save_json(summary, "summary.json", output_dir)
    save_available_ablation_summaries(output_dir)


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
