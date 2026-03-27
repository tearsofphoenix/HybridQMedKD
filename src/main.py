import os
from src.utils.seed import set_seed
from src.utils.io import save_metrics_csv, save_config_json
from src.utils.metrics import aggregate_fold_metrics
from src.trainers.train_teacher import train_teacher_cv
from src.trainers.train_student import run_student_cv

CSV_PATH = "data/raw/wdbc.csv"
PCA_DIM = 4
N_QUBITS = 4
N_SPLITS = 5
SEED = 42
OUTPUT_DIR = "outputs"


def main():
    set_seed(SEED)
    os.makedirs(f"{OUTPUT_DIR}/tables", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)

    print("=" * 60)
    print("Step 1: Train Teacher")
    print("=" * 60)
    teacher_folds = train_teacher_cv(
        csv_path=CSV_PATH,
        pca_dim=PCA_DIM,
        n_splits=N_SPLITS,
        seed=SEED
    )

    experiments = [
        dict(exp_name="student_classic",     model_type="classic", use_kd=False),
        dict(exp_name="student_classic_kd",  model_type="classic", use_kd=True,  alpha=0.5, T=2.0),
        dict(exp_name="student_hybrid_nokd", model_type="hybrid",  use_kd=False, quantum_position="middle"),
        dict(exp_name="student_hybrid_kd",   model_type="hybrid",  use_kd=True,  alpha=0.5, T=2.0, quantum_position="middle"),
    ]

    all_results = {}
    for exp in experiments:
        print("\n" + "=" * 60)
        print(f"Step: {exp['exp_name']}")
        print("=" * 60)
        metrics = run_student_cv(
            csv_path=CSV_PATH,
            teacher_fold_outputs=teacher_folds if exp["use_kd"] else None,
            model_type=exp["model_type"],
            use_kd=exp["use_kd"],
            alpha=exp.get("alpha", 0.5),
            T=exp.get("T", 2.0),
            pca_dim=PCA_DIM,
            n_qubits=N_QUBITS,
            quantum_position=exp.get("quantum_position", "middle"),
            n_splits=N_SPLITS,
            seed=SEED,
            exp_name=exp["exp_name"]
        )
        all_results[exp["exp_name"]] = aggregate_fold_metrics(metrics)
        save_metrics_csv(
            metrics,
            f"{OUTPUT_DIR}/tables/{exp['exp_name']}_folds.csv"
        )

    save_config_json(all_results, f"{OUTPUT_DIR}/tables/summary.json")
    print("\nAll experiments done. Results saved to outputs/tables/")


if __name__ == "__main__":
    main()
