"""CNN direction-classification branch: fit and evaluate the Conv2D model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from volnn.config import Config
from volnn.data.prepare import load_dataset
from volnn.models.builders import build
from volnn.models.common import chronological_split, make_dataset, standard_callbacks
from volnn.utils.io import ensure_dir, save_metrics


def _with_channel(x: np.ndarray) -> np.ndarray:
    """Add the trailing channel axis expected by Conv2D."""
    return x if x.ndim == 4 else x[..., np.newaxis]


def train(cfg: Config, dataset=None) -> dict[str, Any]:
    """Train the CNN classifier and persist model, history and metrics."""
    from tensorflow import keras

    dataset = dataset or load_dataset(cfg)
    x_train, x_val, y_train, y_val = chronological_split(
        dataset.x, dataset.y, cfg.features.split_fraction
    )
    x_train, x_val = _with_channel(x_train), _with_channel(x_val)

    model = build(
        "cnn",
        input_shape=(x_train.shape[1], x_train.shape[2]),
        dropout=cfg.train.dropout,
        l2=cfg.train.l2,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.train.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    name = f"cnn_{cfg.features.tag()}"
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
    """Evaluate the CNN classifier: accuracy, precision, recall, ROC-AUC."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    from tensorflow import keras

    dataset = dataset or load_dataset(cfg)
    _, x_val, _, y_val = chronological_split(dataset.x, dataset.y, cfg.features.split_fraction)
    x_val = _with_channel(x_val)

    model_path = Path(cfg.train.artifacts_dir) / "models" / f"cnn_{cfg.features.tag()}.keras"
    model = keras.models.load_model(str(model_path))
    proba = model.predict(x_val).reshape(-1)
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "precision": float(precision_score(y_val, preds, zero_division=0)),
        "recall": float(recall_score(y_val, preds, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_val, proba))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics
