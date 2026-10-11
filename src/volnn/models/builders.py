"""Keras model builders — architecture mirrors the original project."""

from __future__ import annotations


def build_rnn(input_shape: tuple[int, int], dropout: float = 0.25, l2: float = 1e-5):
    """Build the bidirectional-LSTM regression model.

    Architecture (unchanged from the original ``rnn.py``):
    BiLSTM(128) -> LSTM(128) -> LSTM(64) -> LSTM(32) -> Dense(32) -> Dense(1).
    """
    from keras import Input, regularizers
    from keras.layers import LSTM, Bidirectional, Dense, Dropout
    from keras.models import Sequential

    reg = regularizers.l2(l2)
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=reg)))
    model.add(Dropout(dropout))
    model.add(LSTM(128, return_sequences=True, kernel_regularizer=reg))
    model.add(Dropout(dropout))
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=reg))
    model.add(Dropout(dropout))
    model.add(LSTM(32, kernel_regularizer=reg))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))
    return model


def build_cnn(input_shape: tuple[int, int], dropout: float = 0.5, l2: float = 1e-5):
    """Build the Conv2D binary-classification model.

    ``input_shape`` is ``(past, n_features)``; a channel axis is added
    internally so the conv stack sees ``(past, n_features, 1)``.
    """
    from keras import Input, regularizers
    from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from keras.models import Sequential

    reg = regularizers.l2(l2)
    model = Sequential()
    model.add(Input(shape=(*input_shape, 1)))
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=reg))
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=reg))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.75))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=reg))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=reg))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(dropout))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=reg))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=reg))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(dropout))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=reg))
    model.add(Dropout(dropout))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu", kernel_regularizer=reg))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu", kernel_regularizer=reg))
    model.add(Dense(1, activation="sigmoid"))
    return model


def build(model: str, input_shape: tuple[int, int], dropout: float = 0.25, l2: float = 1e-5):
    """Dispatch to the requested architecture."""
    if model == "cnn":
        return build_cnn(input_shape, dropout=dropout, l2=l2)
    if model == "rnn":
        return build_rnn(input_shape, dropout=dropout, l2=l2)
    raise ValueError(f"unknown model '{model}' (expected 'rnn' or 'cnn')")
