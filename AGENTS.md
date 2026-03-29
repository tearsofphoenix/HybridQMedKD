# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the runnable code: `datasets/` for loading and preprocessing WDBC data, `models/` for teacher/student networks and the quantum layer, `losses/` for distillation losses, `trainers/` for cross-validation training/evaluation, `analysis/` for result aggregation/plotting, and `utils/` for seeds, metrics, and I/O helpers. Experiment settings live in `configs/*.yaml`. Batch runs live in `scripts/`. Use `notebooks/` for exploratory analysis only, and `paper/main.tex` for manuscript updates. Raw data is expected at `data/raw/wdbc.csv`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create and activate a local environment.
- `pip install -r requirements.txt` — install Python dependencies.
- `python src/main.py` — run the main experiment suite; outputs are written under `outputs/tables` and `outputs/logs`.
- `bash scripts/run_all.sh` — wrapper for the default end-to-end run.
- `bash scripts/run_ablation_pca.sh` — run PCA-dimension ablations.
- `bash scripts/run_ablation_position.sh` — run quantum-layer position ablations.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for modules/functions/files, and `PascalCase` for classes such as `StudentHybrid` and `QuantumLayer`. Keep imports grouped as standard library, third-party, then local `src.*` imports. Prefer small, single-purpose functions and keep tensor-shape or training-assumption comments brief and precise.

## Testing Guidelines
There is no committed `tests/` suite yet, so every change must include a local smoke check. At minimum, rerun `python src/main.py` or the relevant ablation script for the code path you touched. For reusable utilities, add deterministic tests under a new `tests/` directory (example: `tests/test_metrics.py`) and keep seeds fixed at `42`. No CI coverage gate is configured; document what you validated in the PR.

## Commit & Pull Request Guidelines
Match the repository’s commit style: short, imperative subjects with prefixes like `fix:`, `docs:`, or `init:`. Keep each commit focused on one experiment, bug fix, or documentation update. PRs should summarize the research goal, list changed configs/scripts, note any data assumptions, and point reviewers to generated artifacts (for example `outputs/tables/summary.json` or updated notebook figures). Include screenshots only when notebook visuals or plots changed.

## Data & Output Hygiene
Do not commit generated artifacts or local datasets. `.gitignore` already excludes `data/raw/`, processed CSVs, model checkpoints, and `outputs/logs/`; keep large experiment outputs outside version control unless they are essential reproducibility assets.
