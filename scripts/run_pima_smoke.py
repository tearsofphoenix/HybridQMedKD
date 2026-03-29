import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.aggregate_results import build_summary_from_csvs, save_summary
from src.analysis.teacher_stats import save_teacher_reports
from src.trainers.train_student import run_student_cv
from src.trainers.train_teacher import train_teacher_cv
from src.utils.seed import set_seed
from src.utils.io import resolve_repo_path


def main():
    set_seed(42)
    csv_path = resolve_repo_path("data", "raw", "pima.csv")

    teacher = train_teacher_cv(
        csv_path,
        dataset_name="pima",
        target_col="class",
        pca_dim=4,
        n_splits=2,
        epochs=5,
        batch_size=32,
    )
    save_teacher_reports(teacher, prefix="teacher_pima_smoke")

    run_student_cv(
        csv_path,
        dataset_name="pima",
        target_col="class",
        model_type="classic",
        use_kd=False,
        pca_dim=4,
        n_splits=2,
        epochs=5,
        batch_size=32,
        exp_name="pima_student_classic_smoke",
    )

    run_student_cv(
        csv_path,
        dataset_name="pima",
        target_col="class",
        teacher_fold_outputs=teacher,
        model_type="hybrid",
        use_kd=True,
        alpha=0.5,
        T=2.0,
        quantum_position="tail",
        pca_dim=4,
        n_splits=2,
        epochs=5,
        batch_size=8,
        exp_name="pima_student_hybrid_kd_smoke",
    )

    summary = build_summary_from_csvs(
        [
            "pima_student_classic_smoke",
            "pima_student_hybrid_kd_smoke",
        ]
    )
    save_summary(summary, output_dir=resolve_repo_path("outputs", "pima_smoke"))


if __name__ == "__main__":
    main()
