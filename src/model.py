"""
model.py
=========

Defines the CNN-BiLSTM architecture used to classify emotions from the
standardized acoustic feature vectors produced by ``feature_extraction.py``
(MFCC + deltas + chroma + mel spectrogram + spectral contrast + tonnetz +
ZCR + RMS, FEATURE_DIM = 275).

Architecture (v5 - reverted to CNN-BiLSTM; Attention removed)
------------------------------------------------------------------
Input(275, 1)
  -> Conv1D(64)  -> BatchNormalization -> MaxPooling1D -> Dropout(0.3)
  -> Conv1D(128) -> BatchNormalization -> MaxPooling1D -> Dropout(0.3)
  -> Bidirectional(LSTM(128, return_sequences=False))
  -> Dense(256, L2) -> Dropout(0.5)
  -> Dense(128, L2) -> Dropout(0.3)
  -> Dense(8, softmax)

Why the Attention layer was removed
--------------------------------------
The previous version (v4) added ``LSTM(return_sequences=True)`` +
``Attention()([x, x])`` + ``GlobalAveragePooling1D()`` between the BiLSTM
and the dense head. In practice this REDUCED test accuracy (81% -> 77.7%)
while train accuracy stayed around 90%, i.e. it widened the train/val gap
rather than narrowing it. With only ~8,640 augmented samples and 69
attention positions to weight, the attention mechanism had enough freedom
to fit spurious positional patterns in the training split that didn't
generalize -- added capacity without added regularization, on a dataset
this size, made overfitting worse rather than better. Reverting to the
simpler ``return_sequences=False`` BiLSTM (which forces the network to
compress the sequence into a single summary vector, rather than letting
it cherry-pick how to weight 69 positions) removes that overfitting lever
entirely.

Architecturally, this file is now line-for-line equivalent to the
81%-accuracy version. The generalization improvements requested are
applied instead in ``train.py`` (Mixup, SpecAugment-style masking, wider
batch size, slower learning rate, longer patience) and
``feature_extraction.py`` (masking helpers), per the "modify training/
augmentation, not architecture" approach since the architecture itself was
not the source of the regression.

Compatibility notes
---------------------
* ``build_model(input_shape, ...)`` keeps the same call signature as
  previous versions, so ``train.py`` (which calls
  ``build_model(input_shape=input_shape)`` generically) requires no
  changes on account of this file alone.
* Input shape remains ``(275, 1)``; output remains an 8-class softmax.
* Every layer used is a standard built-in Keras layer, so the saved
  ``.keras`` file loads via a plain ``tf.keras.models.load_model(path)``
  call with no ``custom_objects`` needed.
* Compilation (Adam, label-smoothed categorical cross-entropy, accuracy
  metric) is unchanged in form; ``train.py`` now passes
  ``learning_rate=5e-4`` explicitly (previously 1e-3).

Author: Senior ML Engineering Team
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

#: Default number of output classes (the 8 RAVDESS emotions).
NUM_CLASSES: int = 8

#: Default learning rate for the Adam optimizer. train.py now passes 5e-4
#: explicitly; this default is kept at 1e-3 only for standalone/direct use
#: of build_model() (e.g. the __main__ sanity check below).
LEARNING_RATE: float = 1e-3

#: Default L2 regularization strength applied to the dense head.
L2_REG: float = 1e-4

#: Default label smoothing factor (0.0 disables smoothing).
LABEL_SMOOTHING: float = 0.00

#: Number of units in the Bidirectional LSTM layer (per direction; the
#: layer's output width is 2x this value).
LSTM_UNITS: int = 128


def build_model(
    input_shape: Tuple[int, int],
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    l2_reg: float = L2_REG,
    label_smoothing: float = LABEL_SMOOTHING,
    lstm_units: int = LSTM_UNITS,
) -> Model:
    """
    Build and compile the Conv1D + Bidirectional LSTM emotion recognition
    model (Attention removed; see module docstring for rationale).

    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape of a single input sample, e.g. ``(275, 1)`` for the full
        pooled feature vector with a single channel.
    num_classes : int, default=8
        Number of output emotion classes.
    learning_rate : float, default=1e-3
        Learning rate passed to the Adam optimizer. ``train.py`` overrides
        this to ``5e-4`` when calling ``build_model``.
    l2_reg : float, default=1e-4
        L2 regularization strength applied to the dense head's kernels.
    label_smoothing : float, default=0.05
        Label smoothing factor passed to the categorical cross-entropy
        loss. Set to ``0.0`` to disable.
    lstm_units : int, default=128
        Number of units in each direction of the Bidirectional LSTM layer.

    Returns
    -------
    tf.keras.Model
        A compiled Keras model ready for training. Saving it via
        ``model.save("models/emotion_model.keras")`` and reloading via
        ``tf.keras.models.load_model("models/emotion_model.keras")``
        requires no ``custom_objects``.
    """
    logger.info(
        "Building CNN-BiLSTM model with input_shape=%s, num_classes=%d, "
        "lstm_units=%d, l2_reg=%.1e, label_smoothing=%.2f, learning_rate=%.1e",
        input_shape,
        num_classes,
        lstm_units,
        l2_reg,
        label_smoothing,
        learning_rate,
    )

    inputs = Input(shape=input_shape, name="feature_input")

    # --- Convolutional block 1 ---
    x = Conv1D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv1d_block1",
    )(inputs)
    x = BatchNormalization(name="batch_norm_block1")(x)
    x = MaxPooling1D(pool_size=2, padding="same", name="maxpool_block1")(x)
    x = Dropout(0.3, name="dropout_block1")(x)

    # --- Convolutional block 2 ---
    x = Conv1D(
        filters=128,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv1d_block2",
    )(x)
    x = BatchNormalization(name="batch_norm_block2")(x)
    x = MaxPooling1D(pool_size=2, padding="same", name="maxpool_block2")(x)
    x = Dropout(0.3, name="dropout_block2")(x)

    # --- Bidirectional LSTM ---
    # return_sequences=False: collapse the sequence to a single summary
    # vector (per direction, concatenated) for the dense head. This is the
    # pre-attention behavior that achieved 81% test accuracy.
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=False),
        name="bidirectional_lstm",
    )(x)

    # --- Classification head ---
    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_256",
    )(x)
    x = Dropout(0.5, name="dropout_50")(x)

    x = Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_128",
    )(x)
    x = Dropout(0.3, name="dropout_30")(x)

    outputs = Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="emotion_recognition_cnn_bilstm")

    loss_fn = CategoricalCrossentropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    logger.info("Model built and compiled successfully.")
    model.summary(print_fn=lambda line: logger.info(line))

    return model


if __name__ == "__main__":
    # Quick sanity check: build a model for the full 275-dim feature vector.
    sample_model = build_model(input_shape=(275, 1))
    logger.info("Standalone model.py sanity check complete.")