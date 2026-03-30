import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.aggregate_results import build_summary_from_csvs, save_summary
from src.analysis.teacher_stats import save_teacher_reports
from src.trainers.train_student import run_student_cv
from src.trainers.train_teacher import train_teacher_cv
from src.utils.io import resolve_repo_path
from src.utils.seed import set_seed


def main():
    set_seed(42)

    csv_path = resolve_repo_path("data", "raw", "wdbc.csv")
    output_dir = resolve_repo_path("outputs", "wdbc_10fold")
    summary_exp_names = [
        "wdbc10_student_classic",
        "wdbc10_student_classic_kd",
        "wdbc10_student_hybrid_nokd_tail",
        "wdbc10_student_hybrid_kd_tail",
    ]

    teacher = train_teacher_cv(
        csv_path,
        dataset_name="wdbc",
        pca_dim=4,
        n_splits=10,
        seed=42,
        epochs=80,
        batch_size=32,
    )
    save_teacher_reports(teacher, prefix="teacher_wdbc_10fold", output_dir=output_dir)

    run_student_cv(
        csv_path,
        dataset_name="wdbc",
        model_type="classic",
        use_kd=False,
        pca_dim=4,
        n_splits=10,
        seed=42,
        epochs=80,
        batch_size=32,
        exp_name="wdbc10_student_classic",
    )

    run_student_cv(
        csv_path,
        dataset_name="wdbc",
        teacher_fold_outputs=teacher,
        model_type="classic",
        use_kd=True,
        alpha=0.5,
        T=2.0,
        pca_dim=4,
        n_splits=10,
        seed=42,
        epochs=80,
        batch_size=32,
        exp_name="wdbc10_student_classic_kd",
    )

    run_student_cv(
        csv_path,
        dataset_name="wdbc",
        model_type="hybrid",
        use_kd=False,
        quantum_position="tail",
        pca_dim=4,
        n_splits=10,
        seed=42,
        epochs=80,
        batch_size=32,
        exp_name="wdbc10_student_hybrid_nokd_tail",
    )

    run_student_cv(
        csv_path,
        dataset_name="wdbc",
        teacher_fold_outputs=teacher,
        model_type="hybrid",
        use_kd=True,
        alpha=0.5,
        T=2.0,
        quantum_position="tail",
        pca_dim=4,
        n_splits=10,
        seed=42,
        epochs=80,
        batch_size=32,
        exp_name="wdbc10_student_hybrid_kd_tail",
    )

    summary = build_summary_from_csvs(summary_exp_names)
    save_summary(summary, output_dir=output_dir)

    print("\nWDBC 10-fold summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
