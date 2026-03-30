import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.aggregate_results import build_named_summary_map, save_json
from src.trainers.train_student import run_student_cv
from src.trainers.train_teacher import train_teacher_cv
from src.utils.io import get_tables_dir, resolve_repo_path
from src.utils.seed import set_seed


DATASETS = {
    "wdbc": {
        "csv_path": resolve_repo_path("data", "raw", "wdbc.csv"),
        "dataset_name": "wdbc",
        "target_col": None,
        "pca_dim": 4,
        "n_splits": 5,
        "teacher_batch_size": 32,
        "student_batch_size": 32,
    },
    "pima": {
        "csv_path": resolve_repo_path("data", "raw", "pima.csv"),
        "dataset_name": "pima",
        "target_col": "class",
        "pca_dim": 4,
        "n_splits": 3,
        "teacher_batch_size": 32,
        "student_batch_size": 8,
    },
    "heart": {
        "csv_path": resolve_repo_path("data", "raw", "heart.csv"),
        "dataset_name": "heart",
        "target_col": "class",
        "pca_dim": 4,
        "n_splits": 3,
        "teacher_batch_size": 32,
        "student_batch_size": 8,
    },
}

POSITIONS = ("front", "middle", "tail")


def run_dataset(dataset_key, cfg):
    print(f"\n{'=' * 72}")
    print(f"Dataset placement ablation: {dataset_key}")
    print(f"{'=' * 72}")

    teacher = train_teacher_cv(
        cfg["csv_path"],
        dataset_name=cfg["dataset_name"],
        target_col=cfg["target_col"],
        pca_dim=cfg["pca_dim"],
        n_splits=cfg["n_splits"],
        seed=42,
        epochs=80,
        batch_size=cfg["teacher_batch_size"],
    )

    exp_name_map = {}
    for position in POSITIONS:
        exp_name = f"{dataset_key}_placement_{position}"
        exp_name_map[position] = exp_name
        run_student_cv(
            cfg["csv_path"],
            dataset_name=cfg["dataset_name"],
            target_col=cfg["target_col"],
            teacher_fold_outputs=teacher,
            model_type="hybrid",
            use_kd=True,
            alpha=0.5,
            T=2.0,
            quantum_position=position,
            pca_dim=cfg["pca_dim"],
            n_splits=cfg["n_splits"],
            seed=42,
            epochs=80,
            batch_size=cfg["student_batch_size"],
            exp_name=exp_name,
        )
        partial_summary = build_named_summary_map(exp_name_map, output_dir=get_tables_dir())
        partial_path = save_json(partial_summary, f"placement_{dataset_key}.json", output_dir=get_tables_dir())
        print(f"Checkpoint placement summary: {partial_path}")

    summary = build_named_summary_map(exp_name_map, output_dir=get_tables_dir())
    output_path = save_json(summary, f"placement_{dataset_key}.json", output_dir=get_tables_dir())
    print(f"Saved placement summary: {output_path}")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    set_seed(42)
    selected = tuple(sys.argv[1:]) or tuple(DATASETS.keys())
    all_summaries = {}
    for dataset_key in selected:
        if dataset_key not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        cfg = DATASETS[dataset_key]
        all_summaries[dataset_key] = run_dataset(dataset_key, cfg)

    suffix = "_".join(selected)
    combined_path = Path(get_tables_dir()) / f"placement_cross_dataset_{suffix}.json"
    combined_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nSaved combined placement summary: {combined_path}")


if __name__ == "__main__":
    main()
