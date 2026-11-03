"""Run a reproducible validation-only comparison against the retained baseline."""

from __future__ import annotations

import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score

from src.data_loader import DEFAULT_SCALER_PATH, load_scaler
from src.feature_extraction import EMOTION_LABELS, load_features
from src.train import train_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "experiments"
BASELINE = ROOT / "outputs" / "full_run" / "models" / "best_model.keras"


def validation_metrics(model: tf.keras.Model, x_val: np.ndarray, y_val: np.ndarray) -> dict:
    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(np.mean(y_true == y_pred)),
        "classification_report": classification_report(
            y_true, y_pred, labels=range(len(EMOTION_LABELS)),
            target_names=EMOTION_LABELS, output_dict=True, zero_division=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("no_mixup", "no_class_weights"), default="no_mixup")
    args = parser.parse_args()
    if not BASELINE.is_file():
        raise FileNotFoundError(f"Retained baseline checkpoint missing: {BASELINE}")
    # The cache API returns all splits and the training API prepares them all.
    # Test arrays are discarded, never passed to fit, and never scored.
    x_train, x_val, _x_test, y_train, y_val, _y_test = load_features()
    baseline_scaler = load_scaler(DEFAULT_SCALER_PATH)
    x_val = baseline_scaler.transform(x_val).astype(np.float32)[..., np.newaxis]
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(EMOTION_LABELS))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    baseline_model = tf.keras.models.load_model(BASELINE)
    baseline = validation_metrics(baseline_model, x_val, y_val)
    del baseline_model

    configs = {
        "no_mixup": (False, True, "disable Mixup only; retain class weights, architecture, seed and optimizer settings"),
        "no_class_weights": (True, False, "disable class weighting only; retain Mixup, architecture, seed and optimizer settings"),
    }
    use_mixup, use_class_weights, change = configs[args.variant]
    checkpoint_path = run_dir / f"{args.variant}.keras"
    model, history = train_model(
        epochs=50, batch_size=32, use_mixup=use_mixup, use_class_weights=use_class_weights,
        model_path=checkpoint_path, plots_dir=run_dir / "training",
        scaler_path=run_dir / "scaler.pkl",
    )
    candidate = validation_metrics(model, x_val, y_val)
    result = {
        "run_id": run_id,
        "seed": 42,
        "split": "actor-disjoint-16/4/4",
        "cache_schema": 3,
        "selection_metric": "validation macro F1, evaluated after val_loss early stopping",
        "test_evaluated": False,
        "test_used_for_selection": False,
        "baseline": {"checkpoint": "outputs/full_run/models/best_model.keras", **baseline},
        "experiment": {
            "change": change,
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "epochs_run": len(history.epoch),
            "best_val_loss_epoch": int(np.argmin(history.history["val_loss"]) + 1),
            **candidate,
        },
    }
    (run_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "validation_report.txt").write_text(
        "Baseline validation\n" + classification_report(
            np.argmax(y_val, axis=1),
            np.argmax(tf.keras.models.load_model(BASELINE).predict(x_val, verbose=0), axis=1),
            target_names=EMOTION_LABELS, zero_division=0,
        ) + f"\n{args.variant} validation\n" + classification_report(
            np.argmax(y_val, axis=1), np.argmax(model.predict(x_val, verbose=0), axis=1),
            target_names=EMOTION_LABELS, zero_division=0,
        ), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "variant": args.variant,
                      "baseline_val_macro_f1": baseline["macro_f1"],
                      "candidate_val_macro_f1": candidate["macro_f1"],
                      "epochs": len(history.epoch)}, indent=2))


if __name__ == "__main__":
    main()

