"""
model.py
=========

Defines the CNN-BiLSTM architecture used to classify emotions from standardized
acoustic feature vectors produced by feature_extraction.py.

Architecture
------------
Input(275, 1)
  -> Conv1D(64) -> BatchNormalization -> MaxPooling1D -> Dropout
  -> Conv1D(128) -> BatchNormalization -> MaxPooling1D -> Dropout
  -> Bidirectional(LSTM(128))
  -> Dense(256) -> Dropout
  -> Dense(128) -> Dropout
  -> Dense(8, softmax)

The architecture is intentionally compact enough for the RAVDESS dataset while
still combining local feature learning with sequence modelling.

Author: Vignesh R
"""

from __future__ import annotations

import logging
from typing import Tuple

from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    MaxPooling1D,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES: int = 8
LEARNING_RATE: float = 1e-3
L2_REG: float = 1e-4
LABEL_SMOOTHING: float = 0.00
LSTM_UNITS: int = 128


def build_model(
    input_shape: Tuple[int, int],
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    l2_reg: float = L2_REG,
    label_smoothing: float = LABEL_SMOOTHING,
    lstm_units: int = LSTM_UNITS,
) -> Model:
    """Build and compile the CNN + Bidirectional LSTM emotion classifier."""
    logger.info(
        "Building CNN-BiLSTM model: input_shape=%s, classes=%d, lstm_units=%d, lr=%.1e",
        input_shape, num_classes, lstm_units, learning_rate,
    )

    inputs = Input(shape=input_shape, name="feature_input")

    x = Conv1D(64, kernel_size=5, padding="same", activation="relu", name="conv1d_block1")(inputs)
    x = BatchNormalization(name="batch_norm_block1")(x)
    x = MaxPooling1D(pool_size=2, padding="same", name="maxpool_block1")(x)
    x = Dropout(0.3, name="dropout_block1")(x)

    x = Conv1D(128, kernel_size=5, padding="same", activation="relu", name="conv1d_block2")(x)
    x = BatchNormalization(name="batch_norm_block2")(x)
    x = MaxPooling1D(pool_size=2, padding="same", name="maxpool_block2")(x)
    x = Dropout(0.3, name="dropout_block2")(x)

    x = Bidirectional(LSTM(lstm_units, return_sequences=False), name="bidirectional_lstm")(x)

    x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense_256")(x)
    x = Dropout(0.5, name="dropout_50")(x)
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense_128")(x)
    x = Dropout(0.3, name="dropout_30")(x)

    outputs = Dense(num_classes, activation="softmax", name="emotion_output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="emotion_recognition_cnn_bilstm")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )

    logger.info("Model built and compiled successfully.")
    model.summary(print_fn=lambda line: logger.info(line))
    return model


if __name__ == "__main__":
    build_model(input_shape=(275, 1))
