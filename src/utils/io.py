import os
import csv
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def get_repo_root():
    return str(REPO_ROOT)


def resolve_repo_path(*parts):
    return str(REPO_ROOT.joinpath(*parts))


def get_outputs_dir(*parts):
    return str(OUTPUTS_DIR.joinpath(*parts)) if parts else str(OUTPUTS_DIR)


def get_tables_dir(*parts):
    return str(TABLES_DIR.joinpath(*parts)) if parts else str(TABLES_DIR)


def get_figures_dir(*parts):
    return str(FIGURES_DIR.joinpath(*parts)) if parts else str(FIGURES_DIR)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def save_metrics_csv(results, output_path):
    ensure_dir(os.path.dirname(output_path))
    if not results:
        return
    keys = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def save_config_json(cfg, output_path):
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w") as f:
        json.dump(cfg, f, indent=2)
