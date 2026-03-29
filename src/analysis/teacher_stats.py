import json
from pathlib import Path

import numpy as np

from src.utils.io import get_tables_dir
from src.utils.metrics import aggregate_fold_metrics


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def summarize_teacher_folds(fold_outputs):
    metrics = []
    for fold in fold_outputs:
        record = dict(fold["metrics"])
        record["train_time"] = fold.get("train_time", 0.0)
        record["infer_time"] = fold.get("infer_time", 0.0)
        record["fold"] = fold.get("fold", len(metrics))
        metrics.append(record)
    return aggregate_fold_metrics(metrics)


def compute_teacher_entropy_stats(fold_outputs, split="train", confidence_margin=0.45):
    logits_key = "tr_logits" if split == "train" else "va_logits"
    all_probs = []

    for fold in fold_outputs:
        logits = fold[logits_key].detach().cpu().numpy().reshape(-1)
        probs = np.clip(_sigmoid(logits), 1e-6, 1 - 1e-6)
        all_probs.append(probs)

    probs = np.concatenate(all_probs)
    entropy = -probs * np.log2(probs) - (1 - probs) * np.log2(1 - probs)

    return {
        "split": split,
        "num_samples": int(probs.shape[0]),
        "mean_entropy_bits": float(np.mean(entropy)),
        "std_entropy_bits": float(np.std(entropy)),
        "median_entropy_bits": float(np.median(entropy)),
        "frac_abs_p_minus_0_5_gt_0_45": float(np.mean(np.abs(probs - 0.5) > confidence_margin)),
    }


def save_teacher_reports(fold_outputs, prefix="teacher", output_dir=None):
    output_dir = Path(output_dir or get_tables_dir())
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_teacher_folds(fold_outputs)
    train_entropy = compute_teacher_entropy_stats(fold_outputs, split="train")
    val_entropy = compute_teacher_entropy_stats(fold_outputs, split="val")

    summary_path = output_dir / f"{prefix}_summary.json"
    entropy_path = output_dir / f"{prefix}_entropy.json"

    summary_path.write_text(json.dumps(summary, indent=2))
    entropy_path.write_text(
        json.dumps(
            {
                "train": train_entropy,
                "val": val_entropy,
            },
            indent=2,
        )
    )

    print(f"Saved: {summary_path}")
    print(f"Saved: {entropy_path}")
    return {
        "summary_path": str(summary_path),
        "entropy_path": str(entropy_path),
        "summary": summary,
        "entropy": {
            "train": train_entropy,
            "val": val_entropy,
        },
    }
