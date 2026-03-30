"""
Minimal teacher-strength ablation for WDBC.

Trains 2 additional teacher architectures ([32,16] and [128,64]),
then runs only the hybrid-KD student with each teacher's logits.
The existing [64,32] teacher results from the main experiment serve as the baseline.

Usage:
    python scripts/run_ablation_teacher.py

Estimated runtime: ~50 min (2 teachers ~1 min + 2 hybrid-KD runs ~50 min).
"""
import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.seed import set_seed
from src.utils.io import resolve_repo_path, get_tables_dir, save_metrics_csv
from src.utils.metrics import aggregate_fold_metrics
from src.datasets.load_tabular import load_dataset
from src.datasets.preprocess import FoldPreprocessor
from src.models.teacher_mlp import TeacherMLP
from src.models.student_hybrid import StudentHybrid
from src.losses.distill import kd_loss
from src.trainers.evaluate import evaluate_binary

SEED = 42
PCA_DIM = 4
N_SPLITS = 5
EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 32
ALPHA = 0.5
T = 2.0
CSV_PATH = resolve_repo_path("data", "raw", "wdbc.csv")


def ensure_wdbc_data(csv_path):
    """Download WDBC from UCI repository if not present."""
    if os.path.exists(csv_path):
        return
    print("WDBC data not found, downloading from UCI...")
    import pandas as pd

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    col_names = ["ID", "Diagnosis"] + [f"f{i}" for i in range(1, 31)]
    df = pd.read_csv(url, header=None, names=col_names)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} (shape {df.shape})")
    print(f"Diagnosis values: {df['Diagnosis'].unique()}")

TEACHER_CONFIGS = {
    "weak": [32, 16],
    "strong": [128, 64],
}


def train_teacher_fold(X_tr, y_tr, X_va, input_dim, hidden_dims, epochs, lr, batch_size):
    """Train a single teacher fold with custom architecture."""
    model = TeacherMLP(input_dim=input_dim, hidden_dims=hidden_dims)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_tr_t[idx]).view(-1), y_tr_t[idx])
            loss.backward()
            opt.step()
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        tr_logits = model(X_tr_t)
        X_va_t = torch.tensor(X_va, dtype=torch.float32)
        va_probs = torch.sigmoid(model(X_va_t)).view(-1).cpu().numpy()

    return model, tr_logits.detach(), train_time


def fit_hybrid_kd(X_tr, y_tr, teacher_logits, input_dim, n_qubits=4):
    """Train hybrid-KD student for one fold."""
    model = StudentHybrid(input_dim=input_dim, n_qubits=n_qubits, n_q_layers=1, quantum_position="middle")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = torch.nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    teacher_logits = teacher_logits.detach()

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            logits = model(X_tr_t[idx])
            loss = kd_loss(logits, teacher_logits[idx], y_tr_t[idx], alpha=ALPHA, T=T)
            loss.backward()
            opt.step()
    train_time = time.time() - t0
    return model, train_time


def run_ablation():
    set_seed(SEED)
    ensure_wdbc_data(CSV_PATH)
    X, y, _ = load_dataset(CSV_PATH, dataset_name="wdbc")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(X, y))

    results = {}

    for teacher_name, hidden_dims in TEACHER_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Teacher: {teacher_name} {hidden_dims}")
        print(f"{'='*60}")

        teacher_metrics = []
        student_metrics = []
        teacher_fold_outputs = []

        for fold, (tr_idx, va_idx) in enumerate(fold_splits):
            print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

            # Preprocess
            pre = FoldPreprocessor(pca_dim=PCA_DIM)
            X_tr, y_tr = pre.fit_transform(X[tr_idx], y[tr_idx])
            X_va = pre.transform(X[va_idx])
            y_va = y[va_idx]

            # Train teacher
            t_model, tr_logits, t_time = train_teacher_fold(
                X_tr, y_tr, X_va, X_tr.shape[1], hidden_dims, EPOCHS, LR, BATCH_SIZE
            )
            with torch.no_grad():
                va_probs = torch.sigmoid(t_model(torch.tensor(X_va, dtype=torch.float32))).view(-1).numpy()
            t_metrics = evaluate_binary(y_va, va_probs)
            t_metrics["train_time"] = t_time
            teacher_metrics.append(t_metrics)
            n_params = sum(p.numel() for p in t_model.parameters())
            print(f"  Teacher: AUC={t_metrics['auc']:.4f} F1={t_metrics['f1']:.4f} "
                  f"MCC={t_metrics['mcc']:.4f} Params={n_params} Time={t_time:.1f}s")

            # Train hybrid-KD student
            s_model, s_time = fit_hybrid_kd(X_tr, y_tr, tr_logits, X_tr.shape[1])
            s_model.eval()
            with torch.no_grad():
                s_probs = torch.sigmoid(s_model(torch.tensor(X_va, dtype=torch.float32))).view(-1).numpy()
            s_m = evaluate_binary(y_va, s_probs)
            s_m["train_time"] = s_time
            s_m["fold"] = fold
            student_metrics.append(s_m)
            print(f"  Hybrid-KD: AUC={s_m['auc']:.4f} F1={s_m['f1']:.4f} "
                  f"MCC={s_m['mcc']:.4f} Time={s_time:.1f}s")

        # Aggregate
        results[teacher_name] = {
            "hidden_dims": hidden_dims,
            "n_params": n_params,
            "teacher": aggregate_fold_metrics(teacher_metrics),
            "hybrid_kd": aggregate_fold_metrics(student_metrics),
        }

        for role, metrics_list in [("teacher", teacher_metrics), ("hybrid_kd", student_metrics)]:
            vals = {k: [m[k] for m in metrics_list] for k in ["auc", "f1", "mcc"]}
            print(f"\n[{teacher_name}] {role}: "
                  f"AUC={np.mean(vals['auc']):.4f}±{np.std(vals['auc']):.4f}  "
                  f"F1={np.mean(vals['f1']):.4f}±{np.std(vals['f1']):.4f}  "
                  f"MCC={np.mean(vals['mcc']):.4f}±{np.std(vals['mcc']):.4f}")

    # Save
    out_path = os.path.join(get_tables_dir(), "ablation_teacher.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("TEACHER STRENGTH ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Teacher':<20} {'Params':>7} {'T-AUC':>8} {'S-AUC':>8} {'S-F1':>8} {'S-MCC':>8}")
    print("-" * 60)
    for name, data in results.items():
        h = data["hidden_dims"]
        n = data["n_params"]
        ta = data["teacher"]["auc"]["mean"]
        sa = data["hybrid_kd"]["auc"]["mean"]
        sf = data["hybrid_kd"]["f1"]["mean"]
        sm = data["hybrid_kd"]["mcc"]["mean"]
        print(f"  {str(h):<18} {n:>7} {ta:>8.4f} {sa:>8.4f} {sf:>8.4f} {sm:>8.4f}")
    # Reference: current teacher [64,32] from main paper
    print(f"  {'[64, 32]':<18} {'2433':>7} {'0.9939':>8} {'0.9931':>8} {'0.9395':>8} {'0.9057':>8}  (main paper)")


if __name__ == "__main__":
    run_ablation()
