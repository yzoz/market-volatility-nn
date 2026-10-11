"""Baseline models for comparison against the learned models."""

from __future__ import annotations

import numpy as np


def persistence_baseline(hv: np.ndarray, past: int, future: int) -> np.ndarray:
    """Naive regression baseline: predict the last observed ``hv`` in the window.

    Returns predictions aligned with the same sample count as
    :func:`volnn.data.labels.build_regression_dataset`.
    """
    hv = np.asarray(hv, dtype=float)
    n = len(hv) - past - future
    return np.array([hv[i + past - 1] for i in range(n)], dtype=float)


def majority_baseline(y: np.ndarray) -> np.ndarray:
    """Classification baseline: always predict the majority class."""
    y = np.asarray(y)
    if y.size == 0:
        return y
    values, counts = np.unique(y, return_counts=True)
    return np.full_like(y, values[np.argmax(counts)])


def logistic_regression_baseline(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray
) -> np.ndarray:
    """Classification baseline: logistic regression on flattened windows."""
    from sklearn.linear_model import LogisticRegression

    flat_train = x_train.reshape(x_train.shape[0], -1)
    flat_val = x_val.reshape(x_val.shape[0], -1)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(flat_train, y_train)
    return clf.predict_proba(flat_val)[:, 1]
