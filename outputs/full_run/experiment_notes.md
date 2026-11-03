# Validation-only experiment notes

This file records controlled training comparisons for the actor-disjoint RAVDESS split. Candidate comparison uses validation macro F1; each run's checkpoint and early stopping still monitor validation loss. Neither experiment computed test predictions or used test results for selection.

## Baseline

- Split: actor-disjoint 16/4/4 actors (960/240/240 original recordings).
- Cache: schema v3, training-only augmentation, train-only fitted scaler.
- Model: CNN-BiLSTM, 405,128 parameters, input `(None, 275, 1)`, output `(None, 8)`.
- Baseline checkpoint and reports are retained under `outputs/full_run/`.
- Historical final test result: accuracy 0.4042, macro F1 0.3948, weighted F1 0.3944. A prior smoke test exposed the test set; treat this result as exposed and do not use it for model selection.
- Best validation loss 1.53695 at epoch 8; early stopping at epoch 16. The training/validation gap suggests overfitting.

## Experiments

| ID | Change from baseline | Seed | Best epoch | Validation macro F1 | Notes |
|---|---|---:|---:|---:|---|
| baseline | Existing configuration (Mixup + class weights) | 42 | 8 by validation loss | 0.4069 | Existing checkpoint retained |
| `20261102-223804` | Mixup disabled; all other settings held constant | 42 | 5 by validation loss (13 epochs run) | 0.4172 | Accuracy 0.4333 vs baseline 0.4375; weighted F1 0.4284 vs 0.4277. Neutral and Sad F1 rose; Disgust and Surprised F1 fell. Candidate retained separately; one seed and one validation actor split are not enough to establish a robust gain. Test set was not used. |
| `20261102-224122` | Class weighting disabled; all other settings held constant | 42 | selected by validation loss (16 epochs run) | 0.4085 | Accuracy 0.4583; weighted F1 0.4358. Only a negligible macro-F1 change from baseline. Candidate retained separately. Test set was not evaluated. |

At the time of the initial single-seed comparison, Mixup had the higher validation macro F1 (difference +0.0103), with mixed class-level effects. The subsequent paired three-seed replication did not reproduce that gain consistently. Keep the original baseline as the reference checkpoint; do not claim a no-Mixup improvement. No test score was computed for these validation experiments.

## Multi-seed replication

`run_paired_seed_experiments.py` runs matched Mixup and no-Mixup trainings for seeds 42, 43 and 44 on the same schema-v3 actor-disjoint cache. Each pair keeps class weights and all other training settings fixed. The comparison metric is validation macro F1; checkpointing and early stopping continue to use validation loss as in the existing training protocol. Results are written under a unique `outputs/paired_seed_experiments/<run-id>/` directory. The script reads only the validation arrays for scoring and does not evaluate the test set.

Completed run: `20261102-225131`.

| Seed | Mixup validation macro F1 | No-Mixup validation macro F1 | Paired difference (Mixup − No-Mixup) |
|---:|---:|---:|---:|
| 42 | 0.4431 | 0.4172 | +0.0259 |
| 43 | 0.3887 | 0.3926 | −0.0039 |
| 44 | 0.4152 | 0.3702 | +0.0450 |

Mean paired difference (Mixup minus No-Mixup) was +0.0223; Mixup won in 2 of 3 seeds. This validation-only comparison on one fixed actor split does not establish broad generalization. The original Mixup baseline remains the reference. The per-run scores are in `outputs/paired_seed_experiments/20261102-225131/results.json`; checkpoints are excluded. No test predictions or metrics were generated.



