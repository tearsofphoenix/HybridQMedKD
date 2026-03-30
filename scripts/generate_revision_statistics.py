import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.statistics import compare_record_sets, load_fold_records, save_json, summarize_records
from src.utils.io import get_tables_dir, resolve_repo_path


def summarize_csvs(exp_names):
    tables_dir = Path(get_tables_dir())
    data = {}
    for exp_name in exp_names:
        csv_path = tables_dir / f"{exp_name}_folds.csv"
        if not csv_path.exists():
            continue
        records = load_fold_records(csv_path)
        data[exp_name] = {
            "summary": summarize_records(records),
            "records": records,
        }
    return data


def build_wdbc_10fold_stats():
    exp_names = [
        "wdbc10_student_classic",
        "wdbc10_student_classic_kd",
        "wdbc10_student_hybrid_nokd_tail",
        "wdbc10_student_hybrid_kd_tail",
    ]
    data = summarize_csvs(exp_names)
    if len(data) < 3:
        return {}
    comparisons = {
        "classic_vs_hybrid_kd_tail": compare_record_sets(
            data["wdbc10_student_classic"]["records"],
            data["wdbc10_student_hybrid_kd_tail"]["records"],
        ) if "wdbc10_student_classic" in data and "wdbc10_student_hybrid_kd_tail" in data else {},
        "hybrid_nokd_tail_vs_hybrid_kd_tail": compare_record_sets(
            data["wdbc10_student_hybrid_nokd_tail"]["records"],
            data["wdbc10_student_hybrid_kd_tail"]["records"],
        ) if "wdbc10_student_hybrid_nokd_tail" in data and "wdbc10_student_hybrid_kd_tail" in data else {},
        "classic_vs_classic_kd": compare_record_sets(
            data["wdbc10_student_classic"]["records"],
            data["wdbc10_student_classic_kd"]["records"],
        ) if "wdbc10_student_classic" in data and "wdbc10_student_classic_kd" in data else {},
    }
    return {
        "summaries": {name: payload["summary"] for name, payload in data.items()},
        "comparisons": comparisons,
    }


def build_dataset_main_stats():
    groups = {
        "wdbc_5fold": [
            "student_classic",
            "student_classic_kd",
            "student_hybrid_nokd",
            "student_hybrid_kd",
        ],
        "pima_5fold": [
            "pima_student_classic_full",
            "pima_student_classic_kd_full",
            "pima_student_hybrid_nokd_full",
            "pima_student_hybrid_kd_full",
        ],
        "heart_5fold": [
            "heart_student_classic_full",
            "heart_student_classic_kd_full",
            "heart_student_hybrid_nokd_full",
            "heart_student_hybrid_kd_full",
        ],
    }
    dataset_stats = {}
    for dataset_key, exp_names in groups.items():
        data = summarize_csvs(exp_names)
        if len(data) < 4:
            continue
        dataset_stats[dataset_key] = {
            "summaries": {name: payload["summary"] for name, payload in data.items()},
            "comparisons": {
                "hybrid_nokd_vs_hybrid_kd": compare_record_sets(
                    data[exp_names[2]]["records"],
                    data[exp_names[3]]["records"],
                ),
                "classic_vs_hybrid_kd": compare_record_sets(
                    data[exp_names[0]]["records"],
                    data[exp_names[3]]["records"],
                ),
            },
        }
    return dataset_stats


def build_placement_stats():
    tables_dir = Path(get_tables_dir())
    stats = {}
    for dataset_key in ("wdbc", "pima", "heart"):
        front_path = tables_dir / f"{dataset_key}_placement_front_folds.csv"
        middle_path = tables_dir / f"{dataset_key}_placement_middle_folds.csv"
        tail_path = tables_dir / f"{dataset_key}_placement_tail_folds.csv"
        if not (front_path.exists() and middle_path.exists() and tail_path.exists()):
            continue
        front = load_fold_records(front_path)
        middle = load_fold_records(middle_path)
        tail = load_fold_records(tail_path)
        stats[dataset_key] = {
            "summary": {
                "front": summarize_records(front),
                "middle": summarize_records(middle),
                "tail": summarize_records(tail),
            },
            "comparisons": {
                "front_vs_middle": compare_record_sets(front, middle),
                "front_vs_tail": compare_record_sets(front, tail),
                "middle_vs_tail": compare_record_sets(middle, tail),
            },
        }
    return stats


def main():
    report = {
        "wdbc_10fold": build_wdbc_10fold_stats(),
        "dataset_main": build_dataset_main_stats(),
        "placement": build_placement_stats(),
    }
    output_path = resolve_repo_path("outputs", "tables", "revision_statistics.json")
    save_json(report, output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
