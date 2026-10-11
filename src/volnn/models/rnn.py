"""RNN regression branch: fit and evaluate the LSTM model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from volnn.config import Config
from volnn.data.prepare import load_dataset
from volnn.models.builders import build
from volnn.models.common import chronological_split, make_dataset, standard_callbacks
from volnn.utils.io import ensure_dir, save_metrics


def train(cfg: Config, dataset=None) -> dict[str, Any]:
    """Train the RNN regressor and persist model, history and metrics.

    Returns the Keras ``history.history`` mapping.
    """
    from tensorflow import keras

    dataset = dataset or load_dataset(cfg)
    x_train, x_val, y_train, y_val = chronological_split(
        dataset.x, dataset.y, cfg.features.split_fraction
    )

    model = build(
        "rnn",
        input_shape=(x_train.shape[1], x_train.shape[2]),
        dropout=cfg.train.dropout,
        l2=cfg.train.l2,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.train.learning_rate),
        loss="mse",
        metrics=["mean_absolute_error"],
    )

    name = f"rnn_{cfg.features.tag()}"
    artifacts = ensure_dir(cfg.train.artifacts_dir)
    history = model.fit(
        make_dataset(x_train, y_train, cfg.train.batch_size),
        epochs=cfg.train.epochs,
        validation_data=make_dataset(x_val, y_val, cfg.train.batch_size),
        callbacks=standard_callbacks(model, name, artifacts),
        verbose=1,
    )

    model_path = artifacts / "models" / f"{name}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    save_metrics(artifacts / "reports" / f"{name}_history.json", history.history)
    return history.history


def evaluate(cfg: Config, dataset=None) -> dict[str, float]:
    """Evaluate the RNN regressor on the validation split (MAE / RMSE / MSE)."""
    dataset = dataset or load_dataset(cfg)
    _, x_val, _, y_val = chronological_split(dataset.x, dataset.y, cfg.features.split_fraction)

    from tensorflow import keras

    model_path = Path(cfg.train.artifacts_dir) / "models" / f"rnn_{cfg.features.tag()}.keras"
    model = keras.models.load_model(str(model_path))
    preds = model.predict(x_val).reshape(-1)
    mse = float(np.mean((preds - y_val) ** 2))
    return {"mae": float(np.mean(np.abs(preds - y_val))), "rmse": float(np.sqrt(mse)), "mse": mse}
