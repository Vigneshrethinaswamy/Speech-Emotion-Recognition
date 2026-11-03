# Resume entry: Speech Emotion Recognition

**Speech Emotion Recognition with Deep Learning** | Python, TensorFlow/Keras, Librosa, scikit-learn

- Built an eight-class RAVDESS speech classifier with a 405,128-parameter CNN-BiLSTM and a 275-feature acoustic pipeline.
- Implemented actor-disjoint train/validation/test partitioning before augmentation, training-only waveform augmentation, and train-only feature scaling.
- Ran six paired, validation-only Mixup comparisons across seeds 42–44; Mixup won 2/3 runs with a mean paired validation macro-F1 improvement of +0.0223; results are from one fixed actor split.
- Added explicit seed propagation, cache-integrity checks, checkpointing, and regression coverage; all 7 project tests passed.

**Metric note:** The paired scores are validation macro F1, not test accuracy. A historical test report was exposed by an earlier smoke test and is not presented as an untouched final benchmark. The comparison does not establish broad generalization.

Evidence: `outputs/paired_seed_experiments/20261102-225131/results.json`; protocol in `outputs/full_run/experiment_notes.md`.

