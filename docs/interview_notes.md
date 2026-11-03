# Interview notes: Speech Emotion Recognition

## 60-second explanation

I built an eight-class speech emotion classifier for RAVDESS using pooled acoustic features and a CNN-BiLSTM. A key engineering decision was to split by actor before augmentation, so recordings and augmented variants from a speaker cannot leak across train, validation, and test. I fit the scaler on training features only and use validation data for training control and experiment comparison. I compared Mixup and no-Mixup in matched runs across seeds 42, 43, and 44. Mixup won two of three seeds, with a mean paired validation macro-F1 improvement of +0.0223. The test set was not used for those experiments, and I describe the previously exposed historical test report as non-final.

## Architecture and data flow

1. Parse emotion and actor identifiers from RAVDESS filenames and actor folders.
2. Assign all original recordings from each actor to exactly one 70/15/15 train/validation/test split.
3. Generate waveform augmentation only from training actors.
4. Extract 275 time-mean-pooled features: MFCC, delta and delta-delta MFCC, chroma, Mel bands, spectral contrast, Tonnetz, zero-crossing rate, and RMS energy.
5. Fit a StandardScaler on training features and apply it unchanged elsewhere.
6. Feed `(275, 1)` feature sequences through two Conv1D blocks, a 128-unit bidirectional LSTM, dense layers, and an eight-class softmax head. The model has 405,128 parameters.
7. Use validation loss for checkpointing/early stopping. Compare experiment candidates using validation macro F1.

## Why actor-disjoint splitting matters

A random recording split can put the same speaker in training and evaluation, allowing speaker-specific vocal characteristics to inflate apparent generalization. Grouping by actor is a closer test of performance on unseen speakers. Splitting before augmentation also ensures that transformed copies cannot cross split boundaries.

## Why macro F1

The eight emotion classes do not have equal support in RAVDESS. Macro F1 computes F1 per class and averages the class scores equally, so a few well-supported classes cannot dominate the summary. It is useful for comparison, but should be read with per-class metrics and the small validation set in mind.

## Experiment interpretation

| Seed | Mixup macro F1 | No-Mixup macro F1 | Mixup minus No-Mixup |
|---:|---:|---:|---:|
| 42 | 0.4431 | 0.4172 | +0.0259 |
| 43 | 0.3887 | 0.3926 | −0.0039 |
| 44 | 0.4152 | 0.3702 | +0.0450 |

The mean paired change (Mixup minus No-Mixup) is +0.0223 and Mixup wins twice. The six runs use a fixed actor split, so they measure sensitivity to training randomness on that split, not robustness across speakers or populations. They do not support a performance-improvement claim.

## Likely questions

### Why combine CNN and BiLSTM?

The convolutional layers learn local patterns across adjacent acoustic feature positions. The bidirectional LSTM then summarizes dependencies in the resulting sequence representation. Here, the sequence positions are pooled feature dimensions rather than raw audio frames, which is a limitation of this representation.

### How did you prevent leakage?

I grouped on actor identity before creating augmented examples, kept each actor in one split, generated augmentation only for training actors, and fitted the scaler on training features only. The cache records schema, split metadata, shapes, and a source fingerprint so stale arrays can be rejected.

### Did removing Mixup improve the model?

There is no consistent evidence that it did. On the same fixed split, Mixup won two of three seeds and its mean paired change was +0.0223 macro F1. I report the full results and keep the baseline as the reference rather than cherry-picking one seed.

### What is the final test performance?

I do not claim an untouched final benchmark. An earlier smoke test exposed the held-out test set and produced a historical report of 40.42% accuracy and 0.3948 macro F1. That result is disclosed as historical and was not used for the validation-only paired selection. A new untouched test set or a properly planned repeated group evaluation is needed for a clean benchmark.

### What would you improve next?

I would plan repeated actor-grouped evaluation with the split protocol fixed in advance, report uncertainty and per-class metrics, compare the handcrafted pooled features with frame-level log-Mel inputs, and calibrate confidence before using it in any user-facing setting.

## Reproduction pointers

- Paired runner: `python run_paired_seed_experiments.py`
- Exact run record: `outputs/paired_seed_experiments/20261102-225131/results.json`
- Detailed experiment notes: `outputs/full_run/experiment_notes.md`
- Architecture source: `docs/architecture.mmd`

