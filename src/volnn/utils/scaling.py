"""Feature scaling utilities.

The scaler is deliberately fit on the training split only to avoid look-ahead
leakage, then applied unchanged to the validation split.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class TrainFitScaler:
    """A thin wrapper around :class:`MinMaxScaler` enforcing train-only fitting.

    Parameters
    ----------
    round_to:
        Optional number of decimals to round to (the original scripts used 5 for
        the CNN branch). ``None`` disables rounding.
    """

    def __init__(self, round_to: int | None = None) -> None:
        self.round_to = round_to
        self._scaler = MinMaxScaler()

    def fit(self, train: np.ndarray) -> TrainFitScaler:
        """Fit the scaler on training data only."""
        values = np.asarray(train, dtype=float)
        self._scaler.fit(values.reshape(-1, 1))
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Scale ``data`` using the previously fitted statistics."""
        values = np.asarray(data, dtype=float)
        scaled = self._scaler.transform(values.reshape(-1, 1)).reshape(-1)
        if self.round_to is not None:
            scaled = scaled.round(self.round_to)
        return scaled

    def fit_transform(self, train: np.ndarray) -> np.ndarray:
        """Fit on and transform the training data."""
        return self.fit(train).transform(train)


def scale_series(series: np.ndarray, round_to: int | None = None) -> np.ndarray:
    """Convenience helper matching the legacy per-series scaling behaviour."""
    return TrainFitScaler(round_to=round_to).fit_transform(series)


def scale_frame(
    frame: np.ndarray, split_fraction: float = 0.8, round_to: int | None = None
) -> np.ndarray:
    """Scale every feature column with train-only statistics.

    Each column is fit on the first ``split_fraction`` of rows and then applied
    to the whole frame, so validation data never influences the scaler.
    """
    values = np.asarray(frame, dtype=float)
    split = max(1, int(split_fraction * values.shape[0]))
    out = np.empty_like(values)
    for col in range(values.shape[1]):
        scaler = TrainFitScaler(round_to=round_to).fit(values[:split, col])
        out[:, col] = scaler.transform(values[:, col])
    return out
