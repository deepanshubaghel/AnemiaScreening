"""Model-specific raw-CBC preprocessing exactly as fitted for the training subset.

The distributions are *embedded in each prepared checkpoint*, never supplied by
patients. Input feature order is read from that checkpoint; no raw CSV is needed.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class CBCPreprocessor:
    columns: tuple[str, ...]
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_checkpoint(cls, checkpoint: dict):
        prep = checkpoint.get("tabular_preprocessing") or {}
        cols = tuple(prep.get("columns") or [])
        n = len(cols)
        if n == 0:
            raise ValueError("Model missing CBC feature names")
        keys = ("imputer_statistics", "scaler_mean", "scaler_scale")
        if not all(key in prep for key in keys):
            raise ValueError("Model is missing fitted CBC preprocessing statistics")
        med, mean, scale = (
            np.asarray(prep[key], dtype=np.float64) for key in keys
        )
        if any(x.shape != (n,) or not np.isfinite(x).all() for x in (med, mean, scale)):
            raise ValueError("Model has invalid preprocessing statistics")
        if np.any(scale <= 0):
            raise ValueError("Model has invalid CBC scale values")
        return cls(cols, med, mean, scale)

    def transform(self, raw: dict[str, float | None]) -> torch.Tensor:
        if any(col not in raw for col in self.columns):
            raise ValueError("Missing a required CBC field")
        values = [np.nan if raw[col] is None else float(raw[col]) for col in self.columns]
        x = np.asarray([values], dtype=np.float32)
        if np.isinf(x).any():
            raise ValueError("CBC values must be finite")
        missing = np.isnan(x)
        if missing.any():
            rowidx, colidx = np.where(missing)
            x[rowidx, colidx] = self.median[colidx]
        # Match scikit-learn's in-place StandardScaler transform for float32.
        x -= self.mean
        x /= self.scale
        if not np.isfinite(x).all():
            raise ValueError("Could not normalize CBC values")
        return torch.from_numpy(x.copy())
