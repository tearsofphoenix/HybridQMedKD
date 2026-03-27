import os
import csv
import json
from datetime import datetime


def ensure_dir(path):
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
