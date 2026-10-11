"""Visualisation and report generation.

All figures are written to ``artifacts/reports/`` rather than shown, so the
toolkit is usable in headless/CI environments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from volnn.utils.io import ensure_dir


def _plt():
    """Import matplotlib lazily with a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_series(values: np.ndarray, title: str, path: str | Path) -> Path:
    """Line plot of a 1D series."""
    plt = _plt()
    out = Path(path)
    ensure_dir(out.parent)
    plt.figure(figsize=(11, 4))
    plt.plot(np.asarray(values).reshape(-1))
    plt.title(title)
    plt.xlabel("step")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_history(history: dict, title: str, path: str | Path) -> Path:
    """Training vs validation loss curves."""
    plt = _plt()
    out = Path(path)
    ensure_dir(out.parent)
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    plt.figure(figsize=(8, 4))
    plt.plot(loss, "b", label="train")
    plt.plot(val_loss, "r", label="val")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path) -> Path:
    """Scatter/line comparison of predicted and actual values."""
    plt = _plt()
    out = Path(path)
    ensure_dir(out.parent)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    plt.figure(figsize=(11, 4))
    plt.plot(y_true, label="actual")
    plt.plot(y_pred, label="predicted")
    plt.title("Predicted vs actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path) -> Path:
    """Confusion-matrix heatmap for the classifier."""
    plt = _plt()
    out = Path(path)
    ensure_dir(out.parent)
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    size = int(labels.max()) + 1 if labels.size else 1
    matrix = np.zeros((size, size), dtype=int)
    for t, p in zip(y_true, y_pred, strict=False):
        matrix[t, p] += 1
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion matrix")
    plt.xlabel("predicted")
    plt.ylabel("actual")
    for (i, j), value in np.ndenumerate(matrix):
        plt.text(j, i, str(value), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_roc(y_true: np.ndarray, proba: np.ndarray, path: str | Path) -> Path:
    """ROC curve for the classifier."""
    from sklearn.metrics import roc_curve

    plt = _plt()
    out = Path(path)
    ensure_dir(out.parent)
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), np.asarray(proba).ravel())
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.title("ROC curve")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out
