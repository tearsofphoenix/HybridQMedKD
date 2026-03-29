import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.aggregate_results import save_available_ablation_summaries
from src.trainers.train_student import run_student_cv
from src.trainers.train_teacher import train_teacher_cv
from src.utils.io import resolve_repo_path
from src.utils.seed import set_seed


def main():
    set_seed(42)
    csv_path = resolve_repo_path("data", "raw", "wdbc.csv")

    teacher = train_teacher_cv(
        csv_path,
        pca_dim=4,
        n_splits=5,
        seed=42,
    )

    for alpha in [0.3, 0.5, 0.7]:
        print(f"Running alpha ablation: alpha={alpha}")
        run_student_cv(
            csv_path,
            teacher_fold_outputs=teacher,
            model_type="hybrid",
            use_kd=True,
            alpha=alpha,
            T=2.0,
            pca_dim=4,
            quantum_position="tail",
            n_splits=5,
            seed=42,
            exp_name=f"ablation_alpha_{alpha}",
        )

    save_available_ablation_summaries()


if __name__ == "__main__":
    main()
