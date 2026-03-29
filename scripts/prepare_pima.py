from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "pima.csv"
    sklearn_cache = repo_root / ".cache" / "scikit_learn"
    sklearn_cache.mkdir(parents=True, exist_ok=True)

    dataset = fetch_openml(
        name="diabetes",
        version=1,
        as_frame=True,
        data_home=str(sklearn_cache),
    )
    df = dataset.frame.copy()

    if "class" not in df.columns:
        target_col = dataset.target.name if dataset.target is not None else df.columns[-1]
        df = df.rename(columns={target_col: "class"})

    numeric_class = pd.to_numeric(df["class"], errors="coerce")
    if numeric_class.isna().any():
        classes = list(pd.Series(df["class"]).dropna().unique())
        if len(classes) != 2:
            raise ValueError(f"Expected binary labels for Pima, got: {classes}")
        lowered = {str(c).strip().lower(): c for c in classes}
        if "tested_negative" in lowered and "tested_positive" in lowered:
            label_map = {
                lowered["tested_negative"]: 0,
                lowered["tested_positive"]: 1,
            }
        else:
            label_map = {classes[0]: 0, classes[1]: 1}
        df["class"] = df["class"].map(label_map).astype(int)
    else:
        df["class"] = numeric_class.astype(int)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Positive rate: {df['class'].mean():.4f}")


if __name__ == "__main__":
    main()
