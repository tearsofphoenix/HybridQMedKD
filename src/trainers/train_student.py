import time
import torch
import numpy as np

from src.models.student_classic import StudentClassic
from src.models.student_hybrid import StudentHybrid
from src.losses.distill import kd_loss
from src.trainers.evaluate import evaluate_binary
from src.utils.io import get_tables_dir, save_metrics_csv


def fit_student(
    model,
    X_train,
    y_train,
    teacher_logits=None,
    use_kd=False,
    alpha=0.5,
    T=2.0,
    epochs=80,
    lr=1e-3,
    batch_size=32,
    verbose_every=5
):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if teacher_logits is not None:
        teacher_logits = teacher_logits.detach()

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(X_train_t[idx])
            if use_kd and teacher_logits is not None:
                loss = kd_loss(logits, teacher_logits[idx], y_train_t[idx],
                               alpha=alpha, T=T)
            else:
                loss = bce(logits.view(-1), y_train_t[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % verbose_every == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  elapsed={elapsed:.1f}s")

    train_time = time.time() - t0
    return model, train_time


def run_student_cv(
    csv_path,
    teacher_fold_outputs=None,
    model_type="classic",
    use_kd=False,
    alpha=0.5,
    T=2.0,
    pca_dim=4,
    n_qubits=4,
    n_q_layers=1,
    quantum_position="middle",
    n_splits=5,
    seed=42,
    epochs=80,
    lr=1e-3,
    batch_size=32,
    exp_name="exp"
):
    from src.datasets.load_wdbc import load_wdbc
    from src.datasets.preprocess import FoldPreprocessor
    from sklearn.model_selection import StratifiedKFold

    X, y, _ = load_wdbc(csv_path)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_splits = list(skf.split(X, y))
    all_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(fold_splits):
        print(f"[{exp_name}] Fold {fold + 1}/{n_splits}")

        if teacher_fold_outputs is not None:
            pre = teacher_fold_outputs[fold]["preprocessor"]
            tr_logits = teacher_fold_outputs[fold]["tr_logits"]
            X_tr = pre.transform(X[tr_idx])
            y_tr = y[tr_idx]
        else:
            pre = FoldPreprocessor(pca_dim=pca_dim)
            X_tr, y_tr = pre.fit_transform(X[tr_idx], y[tr_idx])
            tr_logits = None

        X_va = pre.transform(X[va_idx])
        y_va = y[va_idx]

        if model_type == "classic":
            model = StudentClassic(input_dim=X_tr.shape[1])
        else:
            model = StudentHybrid(
                input_dim=X_tr.shape[1],
                n_qubits=n_qubits,
                n_q_layers=n_q_layers,
                quantum_position=quantum_position
            )

        model, train_time = fit_student(
            model, X_tr, y_tr,
            teacher_logits=tr_logits,
            use_kd=use_kd,
            alpha=alpha, T=T,
            epochs=epochs, lr=lr, batch_size=batch_size
        )

        model.eval()
        X_va_t = torch.tensor(X_va, dtype=torch.float32)
        with torch.no_grad():
            t_inf = time.time()
            probs = torch.sigmoid(model(X_va_t)).view(-1).cpu().numpy()
            infer_time = time.time() - t_inf

        metrics = evaluate_binary(y_va, probs)
        metrics["train_time"] = train_time
        metrics["infer_time"] = infer_time
        metrics["fold"] = fold
        all_metrics.append(metrics)

        print(f"  -> AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
              f"MCC={metrics['mcc']:.4f}  Train={train_time:.1f}s")

    for k in ["auc", "f1", "acc", "mcc"]:
        vals = [m[k] for m in all_metrics]
        print(f"[{exp_name}] {k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    if exp_name:
        output_path = get_tables_dir(f"{exp_name}_folds.csv")
        save_metrics_csv(all_metrics, output_path)
        print(f"[{exp_name}] Saved fold metrics to {output_path}")

    return all_metrics
