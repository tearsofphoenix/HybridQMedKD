import time
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.datasets.load_tabular import load_dataset
from src.datasets.preprocess import FoldPreprocessor
from src.models.teacher_mlp import TeacherMLP
from src.trainers.evaluate import evaluate_binary


def train_teacher_cv(
    csv_path,
    dataset_name="wdbc",
    target_col=None,
    id_col=None,
    positive_label=None,
    negative_label=None,
    drop_cols=None,
    pca_dim=4,
    balance_method="none",
    n_splits=5,
    seed=42,
    epochs=80,
    lr=1e-3,
    batch_size=32
):
    """
    Train teacher MLP with cross-validation.
    Returns fold outputs including logits for distillation.
    """
    X, y, _ = load_dataset(
        csv_path,
        dataset_name=dataset_name,
        target_col=target_col,
        id_col=id_col,
        positive_label=positive_label,
        negative_label=negative_label,
        drop_cols=drop_cols,
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_outputs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"[Teacher] Fold {fold + 1}/{n_splits}")

        pre = FoldPreprocessor(pca_dim=pca_dim, balance_method=balance_method)
        X_tr, y_tr = pre.fit_transform(X[tr_idx], y[tr_idx])
        X_va = pre.transform(X[va_idx])
        y_va = y[va_idx]

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        X_va_t = torch.tensor(X_va, dtype=torch.float32)

        model = TeacherMLP(input_dim=X_tr.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        t0 = time.time()
        for epoch in range(epochs):
            model.train()
            # Mini-batch training
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
            t_inf = time.time()
            va_logits = model(X_va_t)
            infer_time = time.time() - t_inf
            va_probs = torch.sigmoid(va_logits).view(-1).cpu().numpy()

        metrics = evaluate_binary(y_va, va_probs)
        print(f"  AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
              f"MCC={metrics['mcc']:.4f}  Time={train_time:.1f}s")

        fold_outputs.append({
            "fold": fold,
            "model": model,
            "preprocessor": pre,
            "tr_logits": tr_logits.detach(),
            "va_logits": va_logits.detach(),
            "tr_idx": tr_idx,
            "va_idx": va_idx,
            "metrics": metrics,
            "train_time": train_time,
            "infer_time": infer_time,
        })

    avg_auc = np.mean([o["metrics"]["auc"] for o in fold_outputs])
    avg_f1 = np.mean([o["metrics"]["f1"] for o in fold_outputs])
    print(f"\n[Teacher] Mean AUC={avg_auc:.4f}  Mean F1={avg_f1:.4f}")
    return fold_outputs
