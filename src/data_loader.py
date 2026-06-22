"""
data_loader.py
===============

Loads the ALREADY-SPLIT acoustic feature arrays (``X_train.npy`` /
``X_test.npy`` / ``y_train.npy`` / ``y_test.npy``) and prepares them for
CNN-BiLSTM training (v4 - leakage-safe; splitting removed from this module).

    1. Load the four pre-split arrays from disk -- the train/test split
       itself is now decided in ``feature_extraction.process_dataset()``,
       on source-file identity, BEFORE augmentation exists. This module no
       longer calls ``train_test_split`` (or any splitter) at all.
    2. Fit a ``StandardScaler`` on the TRAINING split only, then apply the
       same fitted transform to both train and test splits. This avoids a
       SECOND, independent leakage vector: even with a correct file-level
       split, fitting the scaler on combined data would leak test-set
       statistics into the features the model trains on.
    3. Persist the fitted scaler to disk (``models/scaler.pkl``) so that
       ``predict.py`` can apply an identical transform to new audio at
       inference time.
    4. One-hot encode the labels via ``tf.keras.utils.to_categorical``.
    5. Reshape the feature matrices into the 3-D ``(samples, timesteps,
       channels)`` format expected by a ``Conv1D``-based model.

Why splitting moved out of this module
------------------------------------------
Previously, this module loaded one combined ``X``/``y`` array (originals +
augmented rows mixed together) and called ``train_test_split`` on it,
which could place an augmented variant of an utterance in test while the
same utterance's original row (or another augmented variant) landed in
train -- a data leakage path. The fix required moving the split decision
upstream, to before augmentation exists at all, which is necessarily a
``feature_extraction.py`` concern (only that module has access to raw
source files). This module's responsibility is now strictly: load
pre-split arrays, standardize, reshape, encode -- it has no information
about (and makes no decisions about) which source files ended up in which
split.

This module has no side effects on import; call ``load_and_prepare_data()``
to retrieve everything needed for training in one call.

Author: Senior ML Engineering Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    # Support both `python src/data_loader.py` and `from src import data_loader`
    from feature_extraction import (
        DEFAULT_X_TEST_PATH,
        DEFAULT_X_TRAIN_PATH,
        DEFAULT_Y_TEST_PATH,
        DEFAULT_Y_TRAIN_PATH,
        load_features,
    )
except ImportError:  # pragma: no cover
    from src.feature_extraction import (
        DEFAULT_X_TEST_PATH,
        DEFAULT_X_TRAIN_PATH,
        DEFAULT_Y_TEST_PATH,
        DEFAULT_Y_TRAIN_PATH,
        load_features,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

#: Number of distinct emotion classes (Neutral, Calm, Happy, Sad, Angry,
#: Fearful, Disgust, Surprised).
NUM_CLASSES: int = 8

#: Default path for the persisted StandardScaler, fitted on the training
#: split. predict.py loads this to transform new audio identically.
DEFAULT_SCALER_PATH = Path("models/scaler.pkl")


def reshape_for_cnn(X: np.ndarray) -> np.ndarray:
    """
    Reshape a 2-D feature matrix into the 3-D shape required by a
    ``Conv1D`` input layer: ``(n_samples, n_features, 1)``.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Reshaped array of shape ``(n_samples, n_features, 1)``.
    """
    return X.reshape(X.shape[0], X.shape[1], 1).astype(np.float32)


def compute_class_weights(y_train_int: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights from integer training labels, to be
    passed as ``class_weight`` to ``model.fit()`` (or blended into
    per-sample weights, as ``train.py``'s Mixup pipeline does).

    RAVDESS has a built-in class imbalance: "Neutral" only has a "normal"
    intensity variant (no "strong" version like the other seven emotions),
    so it ends up with roughly half as many samples per actor. Without
    correction, the model tends to under-predict (or entirely ignore)
    minority classes such as Neutral. Balanced class weights counteract
    this by penalizing misclassifications of under-represented classes
    more heavily during training.

    Parameters
    ----------
    y_train_int : np.ndarray
        1-D array of integer class labels (not one-hot encoded) for the
        training split.

    Returns
    -------
    Dict[int, float]
        Mapping from integer class label to its training weight.
    """
    classes = np.unique(y_train_int)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_int,
    )
    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}

    logger.info("Computed balanced class weights: %s", class_weight_dict)

    return class_weight_dict


def fit_and_save_scaler(
    X_train_2d: np.ndarray,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
) -> StandardScaler:
    """
    Fit a ``StandardScaler`` on the (2-D) training feature matrix and
    persist it to disk.

    Parameters
    ----------
    X_train_2d : np.ndarray
        Training feature matrix of shape ``(n_train_samples, n_features)``,
        BEFORE reshaping for the CNN. Must be 2-D.
    scaler_path : str or Path, default="models/scaler.pkl"
        Destination path for the persisted scaler.

    Returns
    -------
    StandardScaler
        The fitted scaler (already saved to disk as a side effect).
    """
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    scaler.fit(X_train_2d)

    joblib.dump(scaler, scaler_path)
    logger.info("Fitted StandardScaler on training data and saved to '%s'.", scaler_path)

    return scaler


def load_scaler(scaler_path: str | Path = DEFAULT_SCALER_PATH) -> StandardScaler:
    """
    Load a previously fitted ``StandardScaler`` from disk.

    Parameters
    ----------
    scaler_path : str or Path, default="models/scaler.pkl"
        Path to the persisted scaler.

    Returns
    -------
    StandardScaler
        The loaded scaler.

    Raises
    ------
    FileNotFoundError
        If the scaler file does not exist.
    """
    scaler_path = Path(scaler_path)

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler file not found at '{scaler_path}'. "
            "Run training first so a scaler is fitted and saved."
        )

    scaler = joblib.load(scaler_path)
    logger.info("Loaded StandardScaler from '%s'.", scaler_path)

    return scaler


def load_and_prepare_data(
    x_train_path: str | Path = DEFAULT_X_TRAIN_PATH,
    x_test_path: str | Path = DEFAULT_X_TEST_PATH,
    y_train_path: str | Path = DEFAULT_Y_TRAIN_PATH,
    y_test_path: str | Path = DEFAULT_Y_TEST_PATH,
    num_classes: int = NUM_CLASSES,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    return_class_weights: bool = False,
):
    """
    Load the pre-split (leakage-safe) feature arrays and produce CNN-ready,
    standardized train/test data.

    NOTE: this function no longer performs any train/test splitting. The
    split was already decided in ``feature_extraction.process_dataset()``
    on source-file identity, before augmentation existed. This function's
    job is strictly: load the four pre-split arrays, fit/apply
    standardization, reshape, and one-hot encode.

    Parameters
    ----------
    x_train_path, x_test_path : str or Path
        Paths to the saved pre-split feature matrices.
    y_train_path, y_test_path : str or Path
        Paths to the saved pre-split integer label vectors.
    num_classes : int, default=8
        Number of emotion classes for one-hot encoding.
    scaler_path : str or Path, default="models/scaler.pkl"
        Where to persist the ``StandardScaler`` fitted on the training
        split.
    return_class_weights : bool, default=False
        If True, also compute and return a balanced class-weight dictionary
        derived from the training split's integer labels.

    Returns
    -------
    Tuple
        If ``return_class_weights`` is False (default):
            ``(X_train, X_test, y_train, y_test)`` where the feature arrays
            are standardized (zero mean / unit variance, fit on train only)
            and reshaped to ``(n_samples, n_features, 1)``, and the label
            arrays are one-hot encoded with shape ``(n_samples, num_classes)``.
        If ``return_class_weights`` is True:
            ``(X_train, X_test, y_train, y_test, class_weights)`` with the
            same arrays as above plus a ``Dict[int, float]`` of balanced
            class weights.

    Raises
    ------
    FileNotFoundError
        If any of the four processed feature files do not exist.
    ValueError
        If the loaded arrays are empty or internally mismatched in length.
    """
    X_train, X_test, y_train, y_test = load_features(
        x_train_path=x_train_path,
        x_test_path=x_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
    )

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Mismatched training sample counts: X_train has {X_train.shape[0]} "
            f"rows, y_train has {y_train.shape[0]} rows."
        )
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"Mismatched test sample counts: X_test has {X_test.shape[0]} "
            f"rows, y_test has {y_test.shape[0]} rows."
        )
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Loaded an empty train or test feature array.")

    # --- Standardize: fit on train ONLY, then transform both splits. ---
    # Fitting on the full dataset (or on test data) would leak test-set
    # statistics into training, inflating validation performance.
    scaler = fit_and_save_scaler(X_train, scaler_path=scaler_path)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    class_weights = compute_class_weights(y_train) if return_class_weights else None

    X_train = reshape_for_cnn(X_train)
    X_test = reshape_for_cnn(X_test)

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    logger.info(
        "Data ready. X_train=%s, X_test=%s, y_train=%s, y_test=%s",
        X_train.shape,
        X_test.shape,
        y_train_cat.shape,
        y_test_cat.shape,
    )

    if return_class_weights:
        return X_train, X_test, y_train_cat, y_test_cat, class_weights

    return X_train, X_test, y_train_cat, y_test_cat


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, weights = load_and_prepare_data(
        return_class_weights=True
    )
    logger.info("Standalone data_loader run complete. Class weights: %s", weights)  