"""Run paired validation-only Mixup vs no-Mixup experiments by seed."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from src.data_loader import load_scaler
from src.feature_extraction import DEFAULT_X_VAL_PATH, DEFAULT_Y_VAL_PATH
from src.train import train_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "paired_seed_experiments"


def macro_f1(model: tf.keras.Model, x_val: np.ndarray, y_val: np.ndarray) -> float:
    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # Read only the fixed validation arrays. Training loads the fixed cache,
    # but no test predictions or test metrics are requested or computed.
    x_val = np.load(DEFAULT_X_VAL_PATH, allow_pickle=False).astype(np.float32)
    y_val_int = np.load(DEFAULT_Y_VAL_PATH, allow_pickle=False)
    y_val = tf.keras.utils.to_categorical(y_val_int, num_classes=8)

    paired = []
    for seed in args.seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=False)
        scores = {}
        for variant, use_mixup in (("baseline_mixup", True), ("no_mixup", False)):
            variant_dir = seed_dir / variant
            model, history = train_model(
                epochs=50,
                batch_size=32,
                use_mixup=use_mixup,
                use_class_weights=True,
                seed=seed,
                model_path=variant_dir / "model.keras",
                plots_dir=variant_dir / "training",
                scaler_path=variant_dir / "scaler.pkl",
            )
            scaler = load_scaler(variant_dir / "scaler.pkl")
            x_val_scaled = scaler.transform(x_val).astype(np.float32)[..., np.newaxis]
            scores[variant] = {
                "validation_macro_f1": macro_f1(model, x_val_scaled, y_val),
                "epochs_run": len(history.epoch),
                "checkpoint": str((variant_dir / "model.keras").relative_to(ROOT)),
            }
            del model
        paired.append({
            "seed": seed,
            **scores,
            "macro_f1_difference_mixup_minus_no_mixup": (
                scores["baseline_mixup"]["validation_macro_f1"]
                - scores["no_mixup"]["validation_macro_f1"]
            ),
        })

    results = {
        "run_id": run_id,
        "split": "fixed actor-disjoint schema-v3 cache (16/4/4 actors)",
        "seeds": args.seeds,
        "selection_metric": "validation macro F1 only",
        "test_evaluated": False,
        "test_used_for_selection": False,
        "pairing": "same seed, actor split, class weights, model, optimizer, and training settings; Mixup is the only changed setting",
        "results": paired,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "results": paired}, indent=2))


if __name__ == "__main__":
    main()

