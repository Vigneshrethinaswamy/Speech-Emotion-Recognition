"""
predict.py
===========

Inference utility for running the trained Emotion Recognition model on a
single, arbitrary ``.wav`` file.

Typical usage
-------------
    >>> from predict import predict_emotion
    >>> label, confidence = predict_emotion("path/to/sample.wav")
    >>> print(f"{label} ({confidence:.2%})")
    Happy (87.42%)

Author: Senior ML Engineering Team
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

try:
    from feature_extraction import EMOTION_LABELS, N_MFCC, SAMPLE_RATE, extract_features
    from data_loader import DEFAULT_SCALER_PATH, load_scaler
except ImportError:  # pragma: no cover
    from src.feature_extraction import (
        EMOTION_LABELS,
        N_MFCC,
        SAMPLE_RATE,
        extract_features,
    )
    from src.data_loader import DEFAULT_SCALER_PATH, load_scaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/emotion_model.keras")

# Module-level caches so repeated calls to predict_emotion() in the same
# process don't reload the model/scaler from disk every time.
_cached_model: tf.keras.Model | None = None
_cached_model_path: str | None = None
_cached_scaler = None
_cached_scaler_path: str | None = None


def _load_scaler(scaler_path: str | Path = DEFAULT_SCALER_PATH):
    """
    Load (and cache) the ``StandardScaler`` fitted during training.

    The model was trained on standardized features (zero mean / unit
    variance, fit on the training split only). Applying the identical
    transform at inference time is required -- without it, raw feature
    magnitudes will not match what the model learned to expect, and
    predictions will be unreliable.

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
        If the scaler file does not exist at the given path.
    """
    global _cached_scaler, _cached_scaler_path

    scaler_path = str(scaler_path)

    if _cached_scaler is not None and _cached_scaler_path == scaler_path:
        return _cached_scaler

    _cached_scaler = load_scaler(scaler_path=scaler_path)
    _cached_scaler_path = scaler_path

    return _cached_scaler


def _load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """
    Load (and cache) the trained Keras model from disk.

    Parameters
    ----------
    model_path : str or Path, default="models/emotion_model.keras"
        Path to the saved ``.keras`` model file.

    Returns
    -------
    tf.keras.Model
        The loaded model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the given path.
    """
    global _cached_model, _cached_model_path

    model_path = str(model_path)

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Train the model first by running train.py / main.py."
        )

    if _cached_model is not None and _cached_model_path == model_path:
        return _cached_model

    logger.info("Loading model from '%s'...", model_path)
    _cached_model = tf.keras.models.load_model(model_path)
    _cached_model_path = model_path

    return _cached_model


def predict_emotion(
    file_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
) -> Tuple[str, float]:
    """
    Predict the emotion expressed in a single speech audio file.

    Parameters
    ----------
    file_path : str or Path
        Path to the ``.wav`` file to classify.
    model_path : str or Path, default="models/emotion_model.keras"
        Path to the trained model file.
    scaler_path : str or Path, default="models/scaler.pkl"
        Path to the ``StandardScaler`` fitted during training. The same
        standardization applied to the training data must be applied here,
        or the model's predictions will be unreliable.

    Returns
    -------
    Tuple[str, float]
        ``(predicted_emotion_label, confidence_score)`` where confidence is
        a float in ``[0, 1]`` representing the softmax probability of the
        predicted class.

    Raises
    ------
    FileNotFoundError
        If the audio file, model file, or scaler file does not exist.
    RuntimeError
        If feature extraction fails for the given audio file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: '{file_path}'")

    model = _load_model(model_path=model_path)
    scaler = _load_scaler(scaler_path=scaler_path)

    logger.info("Extracting acoustic features from '%s'...", file_path)
    feature_vector = extract_features(str(file_path), n_mfcc=N_MFCC, sample_rate=SAMPLE_RATE)

    if feature_vector is None:
        raise RuntimeError(f"Failed to extract features from '{file_path}'.")

    # Apply the SAME standardization (fit on the training split) used
    # during training, then reshape to (1, n_features, 1) for the model.
    feature_vector_2d = feature_vector.reshape(1, -1)
    scaled_vector = scaler.transform(feature_vector_2d)
    model_input = scaled_vector.reshape(1, scaled_vector.shape[1], 1).astype(np.float32)

    predictions = model.predict(model_input, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])
    predicted_label = EMOTION_LABELS[predicted_index]

    logger.info(
        "Prediction for '%s': %s (confidence=%.4f)",
        file_path,
        predicted_label,
        confidence,
    )

    return predicted_label, confidence


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for standalone CLI usage."""
    parser = argparse.ArgumentParser(
        description="Predict the emotion expressed in a speech audio file."
    )
    parser.add_argument(
        "wav_path",
        type=str,
        help="Path to the .wav file to classify.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained .keras model file.",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=str(DEFAULT_SCALER_PATH),
        help="Path to the fitted StandardScaler (.pkl) file.",
    )
    return parser


if __name__ == "__main__":
    cli_parser = _build_arg_parser()
    args = cli_parser.parse_args()

    label, score = predict_emotion(
        args.wav_path, model_path=args.model_path, scaler_path=args.scaler_path
    )
    print(f"Predicted Emotion: {label}")
    print(f"Confidence: {score:.2%}")