import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_LABELS = {
    "student_classic":     "S-Classic",
    "student_classic_kd":  "S-Classic-KD",
    "student_hybrid_nokd": "S-Hybrid-NoKD",
    "student_hybrid_kd":   "S-Hybrid-KD",
}
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]


def load_summary(path="outputs/tables/summary.json"):
    with open(path) as f:
        return json.load(f)


def plot_roc_placeholder():
    """
    ROC curve placeholder - replace with actual fold predictions.
    Uses stored fold CSV files to compute ROC per model.
    """
    import pandas as pd
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (key, label) in enumerate(MODEL_LABELS.items()):
        csv_path = f"{OUTPUT_DIR}/tables/{key}_folds.csv"
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if "auc" in df.columns:
            mean_auc = df["auc"].mean()
            # Dummy ROC for illustration; replace with real fold predictions
            ax.plot([0, 0, 1], [0, 1, 1], lw=1.5,
                    color=COLORS[i], linestyle="--",
                    label=f"{label} (AUC={mean_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (5-fold mean AUC)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/roc_curves.pdf", dpi=300)
    fig.savefig(f"{FIGURES_DIR}/roc_curves.png", dpi=300)
    print(f"Saved: {FIGURES_DIR}/roc_curves.png")
    plt.close(fig)


def plot_main_comparison(summary):
    """
    Bar chart comparing AUC / F1 / MCC across model groups.
    """
    models = [k for k in MODEL_LABELS if k in summary]
    labels = [MODEL_LABELS[k] for k in models]
    metrics = ["auc", "f1", "mcc"]
    metric_names = ["AUC", "F1", "MCC"]
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for j, (m, mn) in enumerate(zip(metrics, metric_names)):
        vals = [summary[k][m]["mean"] if m in summary.get(k, {}) else 0 for k in models]
        errs = [summary[k][m]["std"]  if m in summary.get(k, {}) else 0 for k in models]
        ax.bar(x + j * width, vals, width, yerr=errs, label=mn,
               color=COLORS[j], alpha=0.85, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Main Results: AUC / F1 / MCC by Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/main_comparison.pdf", dpi=300)
    fig.savefig(f"{FIGURES_DIR}/main_comparison.png", dpi=300)
    print(f"Saved: {FIGURES_DIR}/main_comparison.png")
    plt.close(fig)


def plot_performance_cost(summary):
    """
    Scatter: training time vs AUC, sized by MCC.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (key, label) in enumerate(MODEL_LABELS.items()):
        if key not in summary:
            continue
        auc_val = summary[key].get("auc", {}).get("mean", 0)
        time_val = summary[key].get("train_time", {}).get("mean", 0)
        mcc_val = summary[key].get("mcc", {}).get("mean", 0.5)
        ax.scatter(time_val, auc_val, s=(mcc_val + 1) * 100,
                   color=COLORS[i], label=label, zorder=3, edgecolors="k", linewidths=0.5)

    ax.set_xlabel("Training Time (s)")
    ax.set_ylabel("AUC")
    ax.set_title("Performance vs Cost\n(bubble size = MCC+1)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/performance_cost.pdf", dpi=300)
    fig.savefig(f"{FIGURES_DIR}/performance_cost.png", dpi=300)
    print(f"Saved: {FIGURES_DIR}/performance_cost.png")
    plt.close(fig)


def plot_ablation_position(results_dict):
    """
    Bar chart for quantum layer position ablation.
    results_dict: {"front": metrics_dict, "middle": ..., "tail": ...}
    """
    positions = ["front", "middle", "tail"]
    metrics = ["auc", "f1", "mcc"]
    metric_names = ["AUC", "F1", "MCC"]
    x = np.arange(len(positions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    for j, (m, mn) in enumerate(zip(metrics, metric_names)):
        vals = [results_dict.get(p, {}).get(m, {}).get("mean", 0) for p in positions]
        errs = [results_dict.get(p, {}).get(m, {}).get("std", 0)  for p in positions]
        ax.bar(x + j * width, vals, width, yerr=errs, label=mn,
               color=COLORS[j], alpha=0.85, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels([p.capitalize() for p in positions])
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Ablation: Quantum Layer Position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/ablation_position.pdf", dpi=300)
    fig.savefig(f"{FIGURES_DIR}/ablation_position.png", dpi=300)
    print(f"Saved: {FIGURES_DIR}/ablation_position.png")
    plt.close(fig)


def plot_ablation_pca(results_dict):
    """
    Line chart: AUC vs PCA dimension.
    results_dict: {4: metrics, 6: metrics, 8: metrics}
    """
    dims = sorted(results_dict.keys())
    auc_vals = [results_dict[d].get("auc", {}).get("mean", 0) for d in dims]
    auc_errs = [results_dict[d].get("auc", {}).get("std", 0)  for d in dims]
    f1_vals  = [results_dict[d].get("f1",  {}).get("mean", 0) for d in dims]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(dims, auc_vals, yerr=auc_errs, marker="o", label="AUC",
                color=COLORS[0], capsize=3)
    ax.plot(dims, f1_vals, marker="s", label="F1", color=COLORS[1])
    ax.set_xlabel("PCA Dimension")
    ax.set_ylabel("Score")
    ax.set_title("Ablation: PCA Input Dimension")
    ax.set_xticks(dims)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/ablation_pca.pdf", dpi=300)
    fig.savefig(f"{FIGURES_DIR}/ablation_pca.png", dpi=300)
    print(f"Saved: {FIGURES_DIR}/ablation_pca.png")
    plt.close(fig)


if __name__ == "__main__":
    summary = load_summary()
    plot_main_comparison(summary)
    plot_performance_cost(summary)
    plot_roc_placeholder()
    print("All figures saved to", FIGURES_DIR)
