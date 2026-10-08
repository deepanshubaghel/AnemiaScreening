"""Developer-only builder. NOT required by the bundled patient-facing app.

Reproduce preprocessing according to v5_1_MAViA_reduced.py from the same-run
patient_split_and_fold_assignment.csv, then embed it in checkpoint dictionaries.
Call this only for preparing a new pair of fold checkpoints.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def build_one(checkpoint_path: Path, split_csv: Path, output: Path):
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = cp["exp_cfg"]
    columns = list(cp["tabular_preprocessing"]["columns"])
    fold = int(cp["fold"])
    df = pd.read_csv(split_csv)
    expected = [*columns, "split", "fold", *(f"{v}_exists" for v in cfg["view_combo"])]
    missing_cols = sorted(set(expected) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Split table missing required columns: {missing_cols}")
    train = df[(df["split"] == "trainval") & (df["fold"] != fold)].copy()
    exists = train[[f"{v}_exists" for v in cfg["view_combo"]]].apply(pd.to_numeric, errors="raise")
    if cfg["sample_policy"] == "complete_selected":
        train = train[exists.eq(1).all(axis=1)]
    elif cfg["sample_policy"] == "missing_aware_selected":
        train = train[exists.sum(axis=1) >= 1]
    else:
        raise ValueError("Unexpected sample policy: " + str(cfg["sample_policy"]))
    if len(train) == 0:
        raise ValueError("Empty training subset")
    train_raw = train[columns].values.astype(np.float32)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(train_raw)
    scaler.fit(train_imputed)
    if len(columns) != len(imputer.statistics_) or len(columns) != len(scaler.mean_):
        raise ValueError("One or more CBC features was entirely missing from training")
    if not all(np.all(np.isfinite(x)) for x in (imputer.statistics_, scaler.mean_, scaler.scale_)):
        raise ValueError("Non-finite fitted preprocessing statistics")
    cp["tabular_preprocessing"] = {
        "columns": columns,
        "imputer_statistics": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "deployment_prepared": True,
        "fitting_n_rows": len(train),
        "fitting_split": "trainval excluding checkpoint validation fold",
        "fitting_image_policy": cfg["sample_policy"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cp, output)
    print(f"[OK] {output.name}: {len(train)} train rows, {len(columns)} features, threshold {cp['best_threshold']:.6f}")
    return imputer, scaler, train_raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--complete", type=Path, required=True)
    parser.add_argument("--missing", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "models")
    args = parser.parse_args()
    build_one(args.complete, args.csv, args.out / "complete_case.pt")
    build_one(args.missing, args.csv, args.out / "missing_aware.pt")
