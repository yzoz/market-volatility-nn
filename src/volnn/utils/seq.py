"""Sliding-window helpers for time-series datasets."""

from __future__ import annotations

import numpy as np


def make_windows(
    features: np.ndarray,
    past: int,
    n_features: int,
) -> np.ndarray:
    """Build non-overlapping-in-index sliding windows of feature rows.

    Parameters
    ----------
    features:
        Array of shape ``(timesteps, n_features)`` (or already 3D windows).
    past:
        Window length (number of timesteps per sample).
    n_features:
        Expected feature count; used as an explicit dimension guard.

    Returns
    -------
    np.ndarray
        Array of shape ``(samples, past, n_features)``.

    Raises
    ------
    ValueError
        If the actual trailing dimension does not match ``n_features`` — this
        guards against the historical "4 channels vs reshape to 7" bug.
    """
    arr = np.asarray(features, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"features must be 2D (timesteps, n_features), got shape {arr.shape}")
    if arr.shape[1] != n_features:
        raise ValueError(
            f"expected {n_features} features, got {arr.shape[1]} — "
            "check FeatureConfig.n_features against the feature builder"
        )
    n_samples = arr.shape[0] - past
    if n_samples <= 0:
        raise ValueError(f"not enough timesteps ({arr.shape[0]}) for a window of {past}")
    out = np.empty((n_samples, past, n_features), dtype=float)
    for i in range(n_samples):
        out[i] = arr[i : i + past]
    return out


def reshape_windows(flat: np.ndarray, n_features: int) -> np.ndarray:
    """Reshape a flattened ``(samples, timesteps * n_features)`` matrix back to 3D.

    The width is derived from ``n_features`` rather than hard-coded, correcting
    the original CNN branch which always divided by 7.
    """
    arr = np.asarray(flat, dtype=float)
    width = arr.shape[1]
    if width % n_features != 0:
        raise ValueError(f"width {width} is not divisible by n_features={n_features}")
    timesteps = width // n_features
    return arr.reshape(arr.shape[0], timesteps, n_features)
