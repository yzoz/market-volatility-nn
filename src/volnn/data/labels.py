"""Target construction for both branches (regression and classification)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    """A supervised dataset with explicit window semantics."""

    x: np.ndarray  # (samples, past, n_features)
    y: np.ndarray  # (samples,) or (samples, 1)


def _percent_change(hv: np.ndarray, i: int, past: int, future: int) -> float:
    now = hv[i + past]
    later = hv[i + past + future]
    return (later - now) / now


def build_regression_dataset(
    frame: np.ndarray,
    hv: np.ndarray,
    past: int,
    future: int,
) -> Dataset:
    """RNN targets: the future value of ``hv``.

    Sample ``i`` uses window ``frame[i:i+past]`` and target ``hv[i+past+future]``,
    matching the original RNN alignment.
    """
    n = len(frame) - past - future
    if n <= 0:
        raise ValueError("not enough timesteps for the requested past/future window")
    x = np.stack([frame[i : i + past] for i in range(n)])
    y = np.array([hv[i + past + future] for i in range(n)], dtype=float)
    return Dataset(x=x, y=y)


def build_classification_dataset(
    frame: np.ndarray,
    hv: np.ndarray,
    past: int,
    future: int,
    threshold: float,
) -> Dataset:
    """CNN targets: direction of the ``hv`` change beyond ``threshold``.

    Only samples whose absolute percent change exceeds ``threshold`` are kept
    (matching the original filtered-window behaviour). Label is ``1`` for an
    upward move and ``0`` for a downward move.
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []
    n = len(frame) - past - future
    if n <= 0:
        raise ValueError("not enough timesteps for the requested past/future window")
    for i in range(n):
        pct = _percent_change(hv, i, past, future)
        if pct >= threshold:
            xs.append(frame[i : i + past])
            ys.append(1)
        elif pct <= -threshold:
            xs.append(frame[i : i + past])
            ys.append(0)
    if not xs:
        raise ValueError(f"no samples passed the threshold {threshold}; lower it or widen the data")
    return Dataset(x=np.stack(xs), y=np.array(ys, dtype=float))
