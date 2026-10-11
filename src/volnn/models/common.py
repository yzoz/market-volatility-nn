"""Shared training helpers: splitting, batching and callbacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def chronological_split(
    x: np.ndarray, y: np.ndarray, split_fraction: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays chronologically (never shuffled) into train/validation."""
    split = int(split_fraction * int(x.shape[0]))
    if split <= 0 or split >= x.shape[0]:
        raise ValueError(
            f"split_fraction={split_fraction} yields an empty split for {x.shape[0]} samples"
        )
    return x[: split - 1], x[split:], y[: split - 1], y[split:]


def make_dataset(x: np.ndarray, y: np.ndarray, batch_size: int):
    """Build a batched ``tf.data.Dataset`` from already-windowed arrays.

    Time order is preserved (no shuffling) so validation remains a genuine
    future hold-out. Using ``tf.data`` avoids Keras ``Sequence`` API churn
    between Keras 2 and 3.
    """
    import tensorflow as tf

    return (
        tf.data.Dataset.from_tensor_slices((x.astype("float32"), y.astype("float32")))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def standard_callbacks(
    model, name: str, artifacts_dir, patience: int = 5, tensorboard: bool = True
):
    """Create the standard callback set used by both branches.

    ``val_loss`` monitoring assumes training is invoked with ``validation_data``.
    """
    from tensorflow import keras

    weights = Path(artifacts_dir) / "weights"
    logs = Path(artifacts_dir) / "logs"
    weights.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=patience),
        keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filepath=str(weights / f"{name}.weights.h5"),
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
        ),
    ]
    if tensorboard and _tensorboard_available():
        callbacks.append(keras.callbacks.TensorBoard(log_dir=str(logs / name)))
    return callbacks


def _tensorboard_available() -> bool:
    """Return True when the optional ``tensorboard`` package is importable."""
    import importlib.util

    return importlib.util.find_spec("tensorboard") is not None
