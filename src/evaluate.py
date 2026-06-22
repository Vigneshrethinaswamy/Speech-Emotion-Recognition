"""
evaluate.py
============

Evaluation pipeline for the trained Emotion Recognition CNN.

Responsibilities
-----------------
    1. Compute overall test accuracy.
    2. Generate a full ``sklearn`` classification report (precision,
       recall, f1-score per class) and save it to disk.
    3. Compute and visualize a confusion matrix using a seaborn heatmap.

Author: Senior ML Engineering Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

try:
    from feature_extraction import EMOTION_LABELS
except ImportError:  # pragma: no cover
    from src.feature_extraction import EMOTION_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_PLOTS_DIR = Path("plots")
DEFAULT_CONFUSION_MATRIX_PATH = DEFAULT_PLOTS_DIR / "confusion_matrix.png"
DEFAULT_CLASSIFICATION_REPORT_PATH = DEFAULT_PLOTS_DIR / "classification_report.txt"


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str] = EMOTION_LABELS,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
) -> Tuple[float, str, np.ndarray]:
    """
    Evaluate a trained model on the held-out test set and persist
    diagnostic artifacts (confusion matrix image + classification report).

    Parameters
    ----------
    model : tf.keras.Model
        The trained Keras model to evaluate.
    X_test : np.ndarray
        Test feature array of shape ``(n_samples, n_features, 1)``.
    y_test : np.ndarray
        One-hot encoded test labels of shape ``(n_samples, n_classes)``.
    class_names : List[str]
        Human-readable names for each class, in label order.
    plots_dir : str or Path, default="plots"
        Directory in which to save the confusion matrix image and
        classification report text file.

    Returns
    -------
    Tuple[float, str, np.ndarray]
        ``(test_accuracy, classification_report_str, confusion_matrix_array)``
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running model.evaluate() on test set (%d samples)...", X_test.shape[0])
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test loss=%.4f, Test accuracy=%.4f", test_loss, test_accuracy)

    logger.info("Generating predictions for classification report / confusion matrix...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    logger.info("Classification report:\n%s", report_str)

    report_path = plots_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Emotion Recognition From Speech - Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report_str)
    logger.info("Saved classification report to '%s'.", report_path)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("Confusion Matrix - Emotion Recognition")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = plots_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix plot to '%s'.", cm_path)

    return test_accuracy, report_str, cm


if __name__ == "__main__":
    try:
        from data_loader import load_and_prepare_data
    except ImportError:  # pragma: no cover
        from src.data_loader import load_and_prepare_data

    logger.info("Loading saved model for standalone evaluation...")
    loaded_model = tf.keras.models.load_model("models/emotion_model.keras")

    _, X_test_data, _, y_test_data = load_and_prepare_data()
    accuracy, report, matrix = evaluate_model(loaded_model, X_test_data, y_test_data)

    logger.info("Standalone evaluate.py run complete. Final accuracy=%.4f", accuracy)