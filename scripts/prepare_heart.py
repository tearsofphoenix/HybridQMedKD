from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "heart.csv"
    sklearn_cache = repo_root / ".cache" / "scikit_learn"
    sklearn_cache.mkdir(parents=True, exist_ok=True)

    dataset = fetch_openml(
        name="heart-statlog",
        version=1,
        as_frame=True,
        data_home=str(sklearn_cache),
    )
    df = dataset.frame.copy()

    if "class" not in df.columns:
        target_col = dataset.target.name if dataset.target is not None else df.columns[-1]
        df = df.rename(columns={target_col: "class"})

    label_map = {
        "absent": 0,
        "present": 1,
    }
    df["class"] = (
        df["class"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(label_map)
    )
    if df["class"].isna().any():
        raise ValueError("Heart labels must map cleanly to {'absent': 0, 'present': 1}.")
    df["class"] = df["class"].astype(int)

    for col in df.columns:
        if col == "class":
            continue
        df[col] = pd.to_numeric(df[col], errors="raise")

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Positive rate: {df['class'].mean():.4f}")


if __name__ == "__main__":
    main()
