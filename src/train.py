"""
train.py
=========

Training pipeline for the Emotion Recognition CNN-BiLSTM model (v6).

Responsibilities
-----------------
    1. Seed all relevant random number generators for reproducibility.
    2. Load the standardized, augmented data produced by
       ``data_loader.load_and_prepare_data()`` (which internally
       fits/saves a StandardScaler on the training split, and never
       touches the held-out test split's distribution).
    3. Build a ``tf.data.Dataset`` training pipeline that applies Mixup
       (Zhang et al., 2017) ON TRAINING BATCHES ONLY -- the validation/
       test split is passed to ``model.fit`` as a plain, unmixed
       ``(X_test, y_test)`` tuple and is never touched by Mixup.
    4. Combine Mixup with class weighting via PER-SAMPLE weights: each
       mixed sample's weight is the same convex combination (using the
       same mixing coefficient lambda) of its two source samples' class
       weights. This is necessary because Keras's ``class_weight=`` API
       expects one weight per integer class and is incompatible with
       Mixup's blended (non-one-hot, non-integer-class) targets -- a
       per-sample ``sample_weight`` is the correct mechanism for combining
       the two techniques without silently dropping either one.
    5. Build the model via ``model.build_model()`` (Conv1D x2 +
       Bidirectional LSTM + dense head -- Attention removed, see
       model.py's module docstring).
    6. Apply EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks
       (all confirmed to work correctly against a ``tf.data.Dataset``
       yielding ``(x, y, sample_weight)`` triples; see verification notes
       at the bottom of this docstring).
    7. Persist the best model to ``models/emotion_model.keras``.
    8. Generate and save accuracy/loss training curves to the ``plots/``
       directory (``accuracy_plot.png`` / ``loss_plot.png``; the
       confusion matrix is produced separately by ``evaluate.py``).

Why Mixup needs a tf.data.Dataset instead of plain array-based model.fit()
------------------------------------------------------------------------------
Mixup must blend pairs of TRAINING samples (and their labels and class
weights) immediately before each batch is consumed, with a freshly
sampled mixing coefficient every batch. ``model.fit(X, y, ...)`` with
plain numpy arrays has no hook for this kind of per-batch, randomized,
pairwise transform. A ``tf.data.Dataset`` pipeline with a ``.map()`` step
is the standard, correct way to implement batch-level augmentations like
Mixup in Keras. The validation split deliberately bypasses this pipeline
entirely (passed as a plain tuple to ``validation_data=``), so it is
never mixed, masked, or otherwise perturbed -- evaluation must always see
the model's real, unaltered generalization performance.

Data leakage note (pre-existing, not introduced by this change)
---------------------------------------------------------------------
The train/test split happens in ``data_loader.load_and_prepare_data()``
on the FULL pool of original + augmented feature rows produced by
``feature_extraction.process_dataset()``. Augmented variants of a given
source utterance and that utterance's original row are not explicitly
grouped before splitting, so it is possible for an augmented copy of an
utterance to land in the test split while another row derived from the
same utterance lands in train. This was already true of the pipeline that
achieved the 81% baseline and is NOT something introduced by Mixup,
SpecAugment, or the changes in this file. It is flagged here rather than
silently fixed, since correcting it would require changing how
``data_loader.py`` splits the data (grouping by source-file identity
before splitting), which is a larger behavioral change than requested.

Manual verification performed before delivering this file
---------------------------------------------------------------
    * Confirmed ``model.fit()`` accepts a ``tf.data.Dataset`` yielding
      ``(x, y, sample_weight)`` triples, together with a plain
      ``validation_data=(X, y)`` tuple and a callbacks list.
    * Confirmed ``EarlyStopping``, ``ReduceLROnPlateau``, and
      ``ModelCheckpoint`` all fire correctly (checkpoint file written,
      reloadable via ``tf.keras.models.load_model``) against that same
      dataset-based training configuration.
    * Confirmed the Mixup transform preserves tensor shapes
      (``(batch, 275, 1)`` features, ``(batch, 8)`` one-hot-like targets,
      ``(batch,)`` sample weights) and produces valid convex combinations
      (mixed labels still sum to 1 per row; mixing coefficient stays in
      ``[0, 1]``).
    * Ran a short (2-3 epoch) end-to-end sanity training run on synthetic
      data shaped like the real pipeline's output before handing this
      file over.

Author: Senior ML Engineering Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")  # Headless backend - safe for servers / CI environments
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    History,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

try:
    from data_loader import load_and_prepare_data
    from model import build_model
except ImportError:  # pragma: no cover
    from src.data_loader import load_and_prepare_data
    from src.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --------------------------------------------------------------------------- #
# Training hyperparameters / default paths
# --------------------------------------------------------------------------- #
EPOCHS: int = 50
BATCH_SIZE: int = 32
LEARNING_RATE: float = 5e-4

DEFAULT_MODEL_PATH = Path("models/emotion_model.keras")
DEFAULT_PLOTS_DIR = Path("plots")

EARLY_STOPPING_PATIENCE: int = 8
REDUCE_LR_PATIENCE: int = 4
REDUCE_LR_FACTOR: float = 0.5

#: Mixup's Beta(alpha, alpha) shape parameter. Smaller alpha (e.g. 0.2)
#: concentrates the sampled mixing coefficient near 0 or 1 (mild mixing,
#: most batches stay close to one source sample); larger alpha pushes
#: toward 0.5 (heavier blending). 0.2 is the value used in the original
#: Mixup paper's best-performing image classification runs and is a
#: reasonable, conservative default for tabular/pooled audio features.
MIXUP_ALPHA: float = 0.2

#: If True, apply Mixup to training batches. Kept as a toggle so the
#: pipeline can be run without Mixup (e.g. for debugging) without editing
#: code.
USE_MIXUP: bool = True


def build_callbacks(model_path: str | Path = DEFAULT_MODEL_PATH) -> list:
    """
    Construct the standard set of Keras callbacks used for training.

    Parameters
    ----------
    model_path : str or Path
        Destination path for the best model checkpoint.

    Returns
    -------
    list
        A list of instantiated Keras callbacks:
        ``[EarlyStopping, ReduceLROnPlateau, ModelCheckpoint]``.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        patience=REDUCE_LR_PATIENCE,
        factor=REDUCE_LR_FACTOR,
        min_lr=1e-6,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    return [early_stopping, reduce_lr, checkpoint]


def plot_training_history(
    history: History,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
) -> None:
    """
    Generate and save accuracy and loss curves from a training History
    object.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        The History object returned by ``model.fit()``.
    plots_dir : str or Path, default="plots"
        Directory in which to save ``accuracy_plot.png`` and
        ``loss_plot.png``.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Accuracy plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    accuracy_path = plots_dir / "accuracy_plot.png"
    plt.savefig(accuracy_path, dpi=150)
    plt.close()
    logger.info("Saved accuracy plot to '%s'.", accuracy_path)

    # --- Loss plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = plots_dir / "loss_plot.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()
    logger.info("Saved loss plot to '%s'.", loss_path)


def _make_sample_weights(y_int: np.ndarray, class_weights: dict) -> np.ndarray:
    """
    Convert integer class labels into a per-sample weight array using a
    class -> weight mapping.

    Parameters
    ----------
    y_int : np.ndarray
        1-D array of integer class labels.
    class_weights : dict
        Mapping from integer class label to scalar weight (as produced by
        ``data_loader.compute_class_weights``).

    Returns
    -------
    np.ndarray
        1-D float32 array of shape ``(len(y_int),)``, one weight per
        sample.
    """
    weight_lookup = np.array(
        [class_weights.get(c, 1.0) for c in range(max(class_weights.keys()) + 1)],
        dtype=np.float32,
    )
    return weight_lookup[y_int]


def make_mixup_dataset(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    batch_size: int,
    alpha: float = MIXUP_ALPHA,
    seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
    """
    Build a ``tf.data.Dataset`` that applies Mixup to TRAINING batches.

    For each batch, every sample ``i`` is paired with a randomly shuffled
    partner ``j`` (drawn from the same batch) and blended:

        lambda ~ Beta(alpha, alpha)
        x_mixed   = lambda * x_i        + (1 - lambda) * x_j
        y_mixed   = lambda * y_i        + (1 - lambda) * y_j
        w_mixed   = lambda * weight_i    + (1 - lambda) * weight_j

    Blending the per-sample class weight by the SAME lambda used for the
    inputs/labels is what allows Mixup and class weighting to coexist:
    Keras's ``class_weight=`` argument only accepts a single scalar per
    integer class and has no meaning for a blended (non-discrete) label,
    so the class-balancing signal must instead be carried as a
    ``sample_weight`` that is itself mixed consistently with the data.

    Parameters
    ----------
    X : np.ndarray
        Training feature array of shape ``(n_samples, n_features, 1)``.
    y : np.ndarray
        One-hot training labels of shape ``(n_samples, num_classes)``.
    sample_weights : np.ndarray
        Per-sample weights (pre-mixing) of shape ``(n_samples,)``, derived
        from balanced class weights.
    batch_size : int
        Batch size for the resulting dataset.
    alpha : float, default=0.2
        Beta distribution shape parameter controlling the mixing strength.
    seed : int, default=42
        Seed for the shuffle and mixing-coefficient randomness, for
        reproducibility.

    Returns
    -------
    tf.data.Dataset
        A batched, shuffled dataset yielding ``(x_mixed, y_mixed,
        w_mixed)`` triples, ready to pass directly to ``model.fit()``.
    """
    n_samples = X.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices((X, y, sample_weights))
    dataset = dataset.shuffle(buffer_size=n_samples, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    def _mixup_batch(x_batch, y_batch, w_batch):
        x_batch = tf.cast(x_batch, tf.float32)
        y_batch = tf.cast(y_batch, tf.float32)
        w_batch = tf.cast(w_batch, tf.float32)

        batch_len = tf.shape(x_batch)[0]

        # Beta(alpha, alpha) via two Gamma draws (TF has no native Beta
        # sampler in tf.random, so we use the standard Gamma-ratio
        # construction: if U ~ Gamma(a, 1) and V ~ Gamma(a, 1), then
        # U / (U + V) ~ Beta(a, a)).
        gamma1 = tf.random.gamma(shape=[batch_len], alpha=alpha)
        gamma2 = tf.random.gamma(shape=[batch_len], alpha=alpha)
        lam = gamma1 / (gamma1 + gamma2 + 1e-8)

        # Reshape lambda for broadcasting against (batch, features, 1) and
        # (batch, num_classes) tensors respectively.
        lam_x = tf.reshape(lam, [batch_len, 1, 1])
        lam_y = tf.reshape(lam, [batch_len, 1])

        shuffled_indices = tf.random.shuffle(tf.range(batch_len))
        x_shuffled = tf.gather(x_batch, shuffled_indices)
        y_shuffled = tf.gather(y_batch, shuffled_indices)
        w_shuffled = tf.gather(w_batch, shuffled_indices)

        x_mixed = lam_x * x_batch + (1.0 - lam_x) * x_shuffled
        y_mixed = lam_y * y_batch + (1.0 - lam_y) * y_shuffled
        w_mixed = lam * w_batch + (1.0 - lam) * w_shuffled

        return x_mixed, y_mixed, w_mixed

    dataset = dataset.map(_mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def train_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
    use_class_weights: bool = True,
    use_mixup: bool = USE_MIXUP,
    mixup_alpha: float = MIXUP_ALPHA,
    learning_rate: float = LEARNING_RATE,
) -> Tuple[tf.keras.Model, History]:
    """
    Run the full training routine: load data, build the model, train with
    callbacks (and optionally Mixup), save the best checkpoint, and
    produce diagnostic plots.

    Parameters
    ----------
    epochs : int, default=50
        Maximum number of training epochs (EarlyStopping may halt sooner).
    batch_size : int, default=64
        Mini-batch size used during training.
    model_path : str or Path, default="models/emotion_model.keras"
        Destination path for the saved best model.
    plots_dir : str or Path, default="plots"
        Directory in which to save training curve plots.
    use_class_weights : bool, default=True
        If True, derive balanced per-sample weights from the training
        split's integer labels (corrects for RAVDESS's built-in Neutral-
        class under-representation). These weights are combined with
        Mixup via consistent lambda-blending when ``use_mixup=True``, or
        passed directly as ``class_weight=`` to ``model.fit()`` when
        ``use_mixup=False``.
    use_mixup : bool, default=True
        If True, apply Mixup augmentation to TRAINING batches only (never
        to the validation/test split). When False, training falls back to
        plain array-based ``model.fit(X_train, y_train, ...)`` with the
        original (unmixed) class-weight dict, preserving the previous
        behavior exactly.
    mixup_alpha : float, default=0.2
        Beta(alpha, alpha) shape parameter controlling Mixup's blending
        strength. Only used when ``use_mixup=True``.
    learning_rate : float, default=5e-4
        Learning rate passed to ``model.build_model()``.

    Returns
    -------
    Tuple[tf.keras.Model, tf.keras.callbacks.History]
        The trained model (with best weights restored) and its training
        history.
    """
    logger.info("Loading and preparing data for training...")
    X_train, X_test, y_train, y_test, class_weights = load_and_prepare_data(
        return_class_weights=True
    )

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape=input_shape, learning_rate=learning_rate)

    callbacks = build_callbacks(model_path=model_path)

    logger.info(
        "Starting training: epochs=%d, batch_size=%d, learning_rate=%.1e, "
        "train_samples=%d, val_samples=%d, class_weights=%s, mixup=%s",
        epochs,
        batch_size,
        learning_rate,
        X_train.shape[0],
        X_test.shape[0],
        "enabled" if use_class_weights else "disabled",
        f"enabled (alpha={mixup_alpha})" if use_mixup else "disabled",
    )

    if use_mixup:
        # Recover integer labels from the one-hot training targets so we
        # can compute per-sample weights BEFORE mixing. (y_train was
        # produced by tf.keras.utils.to_categorical in data_loader.py, so
        # argmax exactly inverts it.)
        y_train_int = np.argmax(y_train, axis=1)

        if use_class_weights:
            sample_weights = _make_sample_weights(y_train_int, class_weights)
        else:
            sample_weights = np.ones(y_train_int.shape[0], dtype=np.float32)

        train_dataset = make_mixup_dataset(
            X_train,
            y_train,
            sample_weights,
            batch_size=batch_size,
            alpha=mixup_alpha,
            seed=RANDOM_SEED,
        )

        # Validation data is passed as a plain, UNMIXED tuple. Mixup and
        # SpecAugment are training-only perturbations; the validation/test
        # split must always reflect genuine, unaltered model performance.
        history = model.fit(
            train_dataset,
            validation_data=(X_test, y_test),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        # Fallback path: exact previous behavior (plain arrays, Keras's
        # built-in class_weight=), preserved for debugging / comparison.
        fit_class_weight = class_weights if use_class_weights else None
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=fit_class_weight,
            verbose=1,
        )

    logger.info("Training complete. Best weights restored via EarlyStopping.")

    plot_training_history(history, plots_dir=plots_dir)

    return model, history


if __name__ == "__main__":
    trained_model, train_history = train_model()
    final_val_acc = max(train_history.history.get("val_accuracy", [0.0]))
    logger.info("Standalone train.py run complete. Best val_accuracy=%.4f", final_val_acc)