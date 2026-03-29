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

    csv_path = resolve_repo_path("data", "raw", "heart.csv")
    output_dir = resolve_repo_path("outputs", "heart_full")
    summary_exp_names = [
        "heart_student_classic_full",
        "heart_student_classic_kd_full",
        "heart_student_hybrid_nokd_full",
        "heart_student_hybrid_kd_full",
    ]

    teacher = train_teacher_cv(
        csv_path,
        dataset_name="heart",
        target_col="class",
        pca_dim=4,
        n_splits=5,
        seed=42,
        epochs=80,
        batch_size=32,
    )
    save_teacher_reports(teacher, prefix="teacher_heart_full", output_dir=output_dir)

    run_student_cv(
        csv_path,
        dataset_name="heart",
        target_col="class",
        model_type="classic",
        use_kd=False,
        pca_dim=4,
        n_splits=5,
        seed=42,
        epochs=80,
        batch_size=32,
        exp_name="heart_student_classic_full",
    )

    run_student_cv(
        csv_path,
        dataset_name="heart",
        target_col="class",
        teacher_fold_outputs=teacher,
        model_type="classic",
        use_kd=True,
        alpha=0.5,
        T=2.0,
        pca_dim=4,
        n_splits=5,
        seed=42,
        epochs=80,
        batch_size=32,
        exp_name="heart_student_classic_kd_full",
    )

    run_student_cv(
        csv_path,
        dataset_name="heart",
        target_col="class",
        model_type="hybrid",
        use_kd=False,
        quantum_position="tail",
        pca_dim=4,
        n_splits=5,
        seed=42,
        epochs=80,
        batch_size=8,
        exp_name="heart_student_hybrid_nokd_full",
    )

    run_student_cv(
        csv_path,
        dataset_name="heart",
        target_col="class",
        teacher_fold_outputs=teacher,
        model_type="hybrid",
        use_kd=True,
        alpha=0.5,
        T=2.0,
        quantum_position="tail",
        pca_dim=4,
        n_splits=5,
        seed=42,
        epochs=80,
        batch_size=8,
        exp_name="heart_student_hybrid_kd_full",
    )

    summary = build_summary_from_csvs(summary_exp_names)
    save_summary(summary, output_dir=output_dir)

    print("\nHeart full summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
