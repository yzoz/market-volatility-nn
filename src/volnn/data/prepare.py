"""Orchestrate feature/label preparation for both model branches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from volnn.config import Config, FeatureConfig
from volnn.data.features import build_feature_frame
from volnn.data.labels import Dataset, build_classification_dataset, build_regression_dataset
from volnn.utils.io import ensure_dir, save_array
from volnn.utils.scaling import scale_frame


def load_klines(path: str | Path) -> pd.DataFrame:
    """Load a raw klines CSV (no header) into a DataFrame."""
    return pd.read_csv(path, header=None)


def feature_arrays(cfg: FeatureConfig, frame: np.ndarray) -> dict[str, np.ndarray]:
    """Split the feature frame into named series for scaling and saving."""
    return {
        "hv": frame[:, 0],
        "ac": frame[:, 1],
        "vol": frame[:, 2],
        "count": frame[:, 3],
    }


def prepare(
    cfg: Config, klines_path: str | Path, out_dir: str | Path | None = None
) -> dict[str, Path]:
    """Build and persist features + labels for the configured branch.

    Returns a mapping of artefact kind -> written path.
    """
    out = Path(out_dir) if out_dir is not None else Path(cfg.data.data_dir)
    ensure_dir(out)

    df = load_klines(klines_path)
    frame = build_feature_frame(df, cfg.features)
    # Labels are derived from the raw, un-scaled hv: the classification
    # threshold is a percentage move and regression targets stay interpretable.
    hv = frame[:, 0]
    # Input features are scaled with train-only statistics (no leakage).
    frame = scale_frame(frame, cfg.features.split_fraction)

    written: dict[str, Path] = {}
    tag = cfg.features.tag()

    if cfg.train.model == "cnn":
        dataset = build_classification_dataset(
            frame, hv, cfg.features.past, cfg.features.future, cfg.features.threshold()
        )
        written["x"] = save_array(out / f"n_{tag}.csv", dataset.x.reshape(dataset.x.shape[0], -1))
        written["y"] = save_array(out / f"f_{tag}.csv", dataset.y)
    else:
        dataset = build_regression_dataset(frame, hv, cfg.features.past, cfg.features.future)
        written["x"] = save_array(
            out / f"n_rnn_{tag}.csv", dataset.x.reshape(dataset.x.shape[0], -1)
        )
        written["y"] = save_array(out / f"f_rnn_{tag}.csv", dataset.y)
    return written


def load_dataset(cfg: Config) -> Dataset:
    """Load a previously prepared dataset from ``data_dir``."""
    tag = cfg.features.tag()
    prefix = "rnn" if cfg.train.model != "cnn" else ""
    n_file = f"n_{prefix + '_' if prefix else ''}{tag}.csv"
    f_file = f"f_{prefix + '_' if prefix else ''}{tag}.csv"
    root = Path(cfg.data.data_dir)
    x = np.loadtxt(root / n_file)
    y = np.loadtxt(root / f_file)
    from volnn.utils.seq import reshape_windows

    if cfg.train.model == "cnn":
        x = reshape_windows(x, cfg.features.n_features)
    else:
        x = x.reshape(x.shape[0], x.shape[1] // cfg.features.n_features, cfg.features.n_features)
    return Dataset(x=x, y=y)
