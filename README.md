# Speech Emotion Recognition

A TensorFlow/Keras deep-learning project that classifies speech recordings into eight emotion categories using acoustic feature engineering and a CNN–Bidirectional LSTM model.

> **Evaluation status:** The current code uses an actor-disjoint train/test split. The earlier approximately 81% result was produced before that evaluation correction and must not be interpreted as performance on unseen speakers. A fresh run with the corrected split is required before reporting a benchmark.

## Overview

The pipeline extracts fixed-length acoustic features from RAVDESS audio and trains a CNN–BiLSTM classifier.

The eight target classes are:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Dataset

The project expects the **RAVDESS audio-speech** dataset locally under `data/ravdess/`.

- 24 actors (speakers)
- 1,440 speech recordings
- 8 emotion labels encoded in the filenames

The dataset is not included in this repository. Download it from its official distribution and place the extracted actor folders under `data/ravdess/`. The extraction code searches recursively for `.wav` files.

## Feature extraction

Features are extracted with Librosa and mean-pooled into a 275-dimensional vector:

- MFCC, delta MFCC and delta-delta MFCC
- Chroma STFT
- Mel spectrogram
- Spectral contrast
- Tonnetz
- Zero-crossing rate
- RMS energy

Training-only waveform augmentation can generate Gaussian-noise, pitch-shifted and time-stretched variants. SpecAugment-style Mel masking is also supported by the extraction functions, but is disabled by default in the current end-to-end pipeline.

## Model

The current model is a one-dimensional CNN followed by a bidirectional LSTM and dense classification layers.

```text
Input: (275, 1)
  → Conv1D(64) → Batch Normalization → Max Pooling → Dropout
  → Conv1D(128) → Batch Normalization → Max Pooling → Dropout
  → Bidirectional LSTM(128)
  → Dense(256) → Dropout
  → Dense(128) → Dropout
  → Dense(8, Softmax)
```

The model is compiled with Adam and categorical cross-entropy. The current model configuration sets label smoothing to 0.0.

## Evaluation protocol: actor-disjoint split

The split is performed on the original recordings **before augmentation**. The group key is the actor directory name (`Actor_01` through `Actor_24`), not the individual audio filename.

- All recordings from a given actor are assigned to either training or test, never both.
- Augmented samples are generated only from training actors.
- Test features are extracted from original, unaugmented recordings.
- The StandardScaler is fitted on training features only and then applied to the test features.
- The default test fraction is 20% of actors, using `GroupShuffleSplit` with random state 42.

This protocol evaluates generalization to speakers not represented in the training split. It is different from a random recording-level split and should not be compared directly with results from that easier protocol.

## Current results and limitations

No verified benchmark for the corrected actor-disjoint split is currently published in this repository. The earlier ~81% figure was obtained before the actor-wise evaluation correction and is therefore not presented as the current model's test accuracy.

To produce a valid result, regenerate the processed arrays with the current actor-wise splitter, train the model, and evaluate it on the held-out actors. Report at least accuracy, macro F1, per-class precision/recall/F1, and the confusion matrix. Because a single 80/20 actor split can be sensitive to which speakers are held out, repeated actor-grouped evaluation would provide a more stable estimate.

## Project structure

```text
Speech-Emotion-Recognition/
├── data/                 # Local RAVDESS data (not committed)
├── data/processed/       # Generated train/test feature arrays
├── models/               # Generated model and scaler artifacts
├── plots/                # Generated training curves and evaluation plots
├── src/
│   ├── feature_extraction.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── main.py
└── requirements.txt
```

## Setup

Python dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the RAVDESS audio-speech actor folders under `data/ravdess/`.

## Run

Regenerate features and split by actor:

```bash
python main.py --force-extract
```

Train/evaluate using cached actor-disjoint features:

```bash
python main.py
```

The pipeline saves the model under `models/` and diagnostic plots/reports under `plots/`. Use `--force-extract` after changing the dataset or split implementation; otherwise existing cached arrays may reflect an older split.

## Tests

The repository includes regression tests for actor-disjoint splitting and rejection of invalid actor-directory paths. Run them with:

```bash
python -m unittest discover -s tests
```

## Reproducibility notes

- Split seed: 42
- Default held-out actor fraction: 20%
- Maximum epochs: 50 (early stopping may stop earlier)
- Training batch size: 32 in `src/train.py`; the command-line entry point currently defaults to 64 and passes that value through.
- Mixup and class weighting are configurable in `src/train.py`; Mixup is enabled by default.
- Processed arrays, model checkpoints and evaluation plots are generated locally and are not included in the repository.

## Technologies

Python, TensorFlow/Keras, Librosa, NumPy, scikit-learn, Matplotlib, Seaborn and Joblib.

## Author

**Vignesh R** — Artificial Intelligence & Machine Learning
