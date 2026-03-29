# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HybridQMedKD is a research project studying hybrid quantum-classical neural networks for biomedical tabular binary classification. A classical teacher MLP distills knowledge into smaller student models — both purely classical and hybrid (with a PennyLane quantum layer). The paper is in `paper/main.tex`.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run default WDBC experiment suite (5-fold CV, all model groups)
python src/main.py

# Run on supplementary datasets
python scripts/run_pima_full.py      # Pima Diabetes
python scripts/run_heart_full.py     # Heart Statlog

# Ablations
bash scripts/run_ablation_pca.sh        # PCA dim: 4/6/8
bash scripts/run_ablation_position.sh   # Quantum position: front/middle/tail
python scripts/run_ablation_alpha.py    # KD alpha: 0.3/0.5/0.7

# Rebuild paper (from paper/)
cd paper && latexmk -pdf main.tex

# Aggregate results from existing fold CSVs
python src/analysis/aggregate_results.py
```

## Architecture

### Experiment Pipeline

`src/main.py` orchestrates: train teacher → run 4 student experiments (classic, classic+KD, hybrid, hybrid+KD) → aggregate results.

Each experiment uses **5-fold stratified cross-validation** with a shared seed (42). Data leakage is prevented by fitting PCA/StandardScaler only on training folds inside `FoldPreprocessor`.

### Key Modules

- **`src/models/`** — `TeacherMLP` (2-layer MLP, 64→32→1), `StudentClassic` (16→8→1), `StudentHybrid` (classical projection → `QuantumLayer` → head), `QuantumLayer` (PennyLane `AngleEmbedding` + `BasicEntanglerLayers`, 4 qubits, parameter-shift gradients)
- **`src/datasets/`** — `load_tabular.py` dispatches to dataset-specific loaders (WDBC, or generic binary CSV). `preprocess.py` has `FoldPreprocessor` (StandardScaler + PCA, optional SMOTE)
- **`src/losses/distill.py`** — `kd_loss()`: weighted combination of BCE hard loss and KL-divergence soft loss with temperature scaling
- **`src/trainers/`** — `train_teacher.py` and `train_student.py` handle CV loops. `train_student.py:run_student_cv()` is the central entry point for all student experiments
- **`src/analysis/`** — Aggregates fold CSVs into JSON summaries, generates plots

### Data Flow

1. CSV loaded via `load_dataset()` → X, y arrays
2. Per fold: `FoldPreprocessor.fit_transform()` on train split, `.transform()` on val split
3. Teacher trains, saves logits per fold → passed to student when `use_kd=True`
4. Student reuses teacher's preprocessor for fair KD comparison
5. Metrics (AUC, F1, ACC, MCC, precision, recall, train/infer time) saved as per-fold CSVs

### Configs

YAML files in `configs/` document hyperparameters but are **not loaded programmatically** — scripts hardcode values directly. When changing experiment settings, update both the config YAML and the script.

### Adding a New Dataset

1. Place CSV in `data/raw/`
2. Call `run_student_cv()` with `dataset_name`, `target_col`, and optionally `id_col`, `positive_label`, `negative_label`, `drop_cols`
3. Or add a dedicated loader in `src/datasets/` and register it in `load_dataset()`

## Conventions

- **Python style**: 4-space indent, `snake_case` for functions/modules, `PascalCase` for classes
- **Import order**: stdlib → third-party → `src.*`
- **Seed**: Always use seed 42 via `src.utils.seed.set_seed()`
- **Quantum circuits**: Keep shallow (4 qubits, 1-2 layers) for local simulation feasibility
- **No formal test suite**: Smoke-check changed code by rerunning `python src/main.py` or the relevant script
- **Commit style**: Short imperative subjects with prefixes (`feat:`, `fix:`, `paper:`, `scripts:`)
- **Do not commit**: `data/raw/`, model checkpoints, `outputs/logs/`, LaTeX build artifacts (all in `.gitignore`)

## Paper

Manuscript at `paper/main.tex`. Figures read from `outputs/figures/` (configured via `\graphicspath`). LaTeX packages (tikz-cd, etc.) are vendored in `.texmf/`. Use `latexmk -pdf` to build.
