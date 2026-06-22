"""
feature_extraction.py
======================

Feature extraction pipeline for the RAVDESS Emotion Recognition project
(v3 - augmentation + expanded feature set).

This module is responsible for:

    1. Parsing emotion labels from RAVDESS filenames.
    2. Generating augmented variants of each training waveform (Gaussian
       noise, pitch shifting, time stretching) to combat RAVDESS's small
       size (1,440 clips) and improve generalization.
    3. Extracting a rich, mean-pooled acoustic feature vector per
       waveform (original + each augmented copy):

         - MFCC (40 coeffs)               : spectral envelope / timbre
         - Delta-MFCC (40 coeffs)         : rate of change of timbre
         - Delta-Delta-MFCC (40 coeffs)   : acceleration of timbre change
         - Chroma STFT (12 bins)          : pitch-class energy distribution
         - Mel Spectrogram (128 bands)    : perceptual frequency energy
         - Spectral contrast (7 bands)    : peak-valley energy contrast
         - Tonnetz (6 dims)               : harmonic/tonal centroid space
         - Zero-crossing rate (1)         : noisiness / voicing
         - RMS energy (1)                 : loudness

       Each block is mean-pooled across time, giving a single fixed-length
       vector (FEATURE_DIM = 375) per waveform.
    4. Walking the entire `data/ravdess/.../Actor_XX/` directory tree to
       build the full feature matrix (X) and label vector (y), expanding
       each file into multiple rows when augmentation is enabled.
    5. Persisting / loading the processed arrays to and from disk as .npy
       files so that downstream scripts (data_loader, train, evaluate)
       never need to touch raw audio again.

RAVDESS filename convention
----------------------------
Filenames follow the pattern:

    03-01-05-01-02-01-01.wav
     |  |  |  |  |  |  |
     |  |  |  |  |  |  └─ Actor ID (01-24, odd = male, even = female)
     |  |  |  |  |  └──── Repetition (01 or 02)
     |  |  |  |  └─────── Statement (01 or 02)
     |  |  |  └────────── Emotional intensity (01 = normal, 02 = strong)
     |  |  └───────────── Emotion (01-08)
     |  └──────────────── Vocal channel (01 = speech, 02 = song)
     └─────────────────── Modality (03 = audio-only)

The third hyphen-separated token is the emotion code, which this module
maps to a zero-indexed integer label using EMOTION_MAP. Note that the
dataset is nested one level deeper than a flat Actor_XX layout in some
distributions (e.g. ``data/ravdess/audio_speech_actors_01-24/Actor_01/``);
``process_dataset`` uses ``rglob`` so this is handled transparently
regardless of nesting depth.

Author: Senior ML Engineering Team
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Mapping from the RAVDESS two-digit emotion code (as found in the filename)
#: to a zero-indexed integer class label used for model training.
EMOTION_MAP: dict[str, int] = {
    "01": 0,  # Neutral
    "02": 1,  # Calm
    "03": 2,  # Happy
    "04": 3,  # Sad
    "05": 4,  # Angry
    "06": 5,  # Fearful
    "07": 6,  # Disgust
    "08": 7,  # Surprised
}

#: Human-readable class names, ordered to match the integer labels above.
EMOTION_LABELS: List[str] = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised",
]

#: Number of MFCC coefficients to extract per audio frame.
N_MFCC: int = 40

#: Number of chroma bins (pitch classes).
N_CHROMA: int = 12

#: Number of Mel Spectrogram bands.
N_MELS: int = 128

#: Number of spectral contrast bands (librosa default produces n_bands + 1 = 7).
N_CONTRAST_BANDS: int = 6

#: Number of Tonnetz (tonal centroid) dimensions - always 6 in librosa.
N_TONNETZ: int = 6

#: Total dimensionality of the final pooled feature vector:
#: MFCC(40) + delta-MFCC(40) + delta2-MFCC(40) + chroma(12) + mel(128)
#: + spectral_contrast(7) + tonnetz(6) + zero_crossing_rate(1) + rms(1) = 275
FEATURE_DIM: int = (
    (N_MFCC * 3)
    + N_CHROMA
    + N_MELS
    + (N_CONTRAST_BANDS + 1)
    + N_TONNETZ
    + 1
    + 1
)

#: Target sample rate used when loading audio (None preserves native rate;
#: we fix it for consistency across the dataset).
SAMPLE_RATE: int = 22050

#: Default locations for raw data and processed feature arrays.
DEFAULT_DATA_DIR = Path("data/ravdess")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_X_TRAIN_PATH = DEFAULT_PROCESSED_DIR / "X_train.npy"
DEFAULT_X_TEST_PATH = DEFAULT_PROCESSED_DIR / "X_test.npy"
DEFAULT_Y_TRAIN_PATH = DEFAULT_PROCESSED_DIR / "y_train.npy"
DEFAULT_Y_TEST_PATH = DEFAULT_PROCESSED_DIR / "y_test.npy"

# --------------------------------------------------------------------------- #
# Augmentation hyperparameters
# --------------------------------------------------------------------------- #

#: Standard deviation (relative to signal amplitude) of additive Gaussian
#: noise used by `augment_noise`.
NOISE_FACTOR: float = 0.005

#: Pitch shift amounts (in semitones) used to generate augmented copies.
PITCH_SHIFT_STEPS: Tuple[float, ...] = (-2.0, 2.0)

#: Time-stretch rate factors used to generate augmented copies
#: (>1.0 = faster/shorter, <1.0 = slower/longer).
TIME_STRETCH_RATES: Tuple[float, ...] = (0.9, 1.1)

#: SpecAugment-style masking parameters, applied to the Mel spectrogram
#: (the one genuinely 2-D time/frequency representation in this pipeline)
#: BEFORE it is mean-pooled into the final feature vector. Following the
#: original SpecAugment paper's naming: a "frequency mask" zeroes a band of
#: mel bins across all time frames; a "time mask" zeroes a block of time
#: frames across all mel bins.
SPEC_AUGMENT_FREQ_MASK_PARAM: int = 12
SPEC_AUGMENT_TIME_MASK_PARAM: int = 8
SPEC_AUGMENT_NUM_FREQ_MASKS: int = 1
SPEC_AUGMENT_NUM_TIME_MASKS: int = 1


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
def parse_emotion_from_filename(filename: str) -> Optional[int]:
    """
    Parse the integer emotion label encoded in a RAVDESS filename.

    Parameters
    ----------
    filename : str
        The audio filename, e.g. ``"03-01-05-01-02-01-01.wav"``. Can be a
        bare filename or a full path; only the basename is used.

    Returns
    -------
    Optional[int]
        The zero-indexed emotion label (0-7) if the filename is well-formed
        and the emotion code is recognized, otherwise ``None``.

    Examples
    --------
    >>> parse_emotion_from_filename("03-01-05-01-02-01-01.wav")
    4
    >>> parse_emotion_from_filename("not-a-valid-file.wav") is None
    True
    """
    basename = os.path.basename(filename)
    stem, _, _ = basename.partition(".")
    parts = stem.split("-")

    if len(parts) < 3:
        logger.warning("Filename '%s' does not match RAVDESS format; skipping.", filename)
        return None

    emotion_code = parts[2]
    label = EMOTION_MAP.get(emotion_code)

    if label is None:
        logger.warning(
            "Unrecognized emotion code '%s' in filename '%s'; skipping.",
            emotion_code,
            filename,
        )
        return None

    return label


# --------------------------------------------------------------------------- #
# Audio augmentation
# --------------------------------------------------------------------------- #
def augment_noise(audio: np.ndarray, noise_factor: float = NOISE_FACTOR) -> np.ndarray:
    """
    Add light Gaussian noise to a waveform.

    Parameters
    ----------
    audio : np.ndarray
        1-D waveform array.
    noise_factor : float, default=0.005
        Standard deviation of the noise, expressed relative to the
        waveform's own amplitude scale.

    Returns
    -------
    np.ndarray
        Noisy copy of ``audio``, same shape and dtype family as the input.
    """
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return augmented.astype(np.float32)


def augment_pitch_shift(
    audio: np.ndarray,
    sample_rate: int,
    n_steps: float,
) -> np.ndarray:
    """
    Shift the pitch of a waveform by a number of semitones, preserving
    duration.

    Parameters
    ----------
    audio : np.ndarray
        1-D waveform array.
    sample_rate : int
        Sample rate of ``audio``.
    n_steps : float
        Number of semitones to shift (positive = higher pitch, negative =
        lower pitch).

    Returns
    -------
    np.ndarray
        Pitch-shifted waveform.
    """
    shifted = librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=n_steps)
    return shifted.astype(np.float32)


def augment_time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    Time-stretch a waveform by a rate factor, preserving pitch.

    Parameters
    ----------
    audio : np.ndarray
        1-D waveform array.
    rate : float
        Stretch factor. Values > 1.0 speed up (shorten) the audio; values
        < 1.0 slow down (lengthen) the audio.

    Returns
    -------
    np.ndarray
        Time-stretched waveform. Length will differ from the input; the
        feature extraction step pools across time so this is handled
        transparently by downstream code.
    """
    stretched = librosa.effects.time_stretch(y=audio, rate=rate)
    return stretched.astype(np.float32)


def generate_augmented_waveforms(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> List[np.ndarray]:
    """
    Generate the full set of augmented waveforms for a single source
    waveform: Gaussian noise, pitch shifts, and time stretches.

    Parameters
    ----------
    audio : np.ndarray
        Original 1-D waveform array.
    sample_rate : int, default=22050
        Sample rate of ``audio``.

    Returns
    -------
    List[np.ndarray]
        List of augmented waveforms (does NOT include the original). One
        entry per augmentation defined by ``NOISE_FACTOR``,
        ``PITCH_SHIFT_STEPS``, and ``TIME_STRETCH_RATES``.
    """
    augmented: List[np.ndarray] = [augment_noise(audio)]

    for steps in PITCH_SHIFT_STEPS:
        try:
            augmented.append(augment_pitch_shift(audio, sample_rate, steps))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pitch-shift augmentation (steps=%s) failed: %s", steps, exc)

    for rate in TIME_STRETCH_RATES:
        try:
            augmented.append(augment_time_stretch(audio, rate))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Time-stretch augmentation (rate=%s) failed: %s", rate, exc)

    return augmented


# --------------------------------------------------------------------------- #
# SpecAugment-style masking (applied to the Mel spectrogram, pre-pooling)
# --------------------------------------------------------------------------- #
def spec_augment_mel(
    mel_db: np.ndarray,
    freq_mask_param: int = SPEC_AUGMENT_FREQ_MASK_PARAM,
    time_mask_param: int = SPEC_AUGMENT_TIME_MASK_PARAM,
    num_freq_masks: int = SPEC_AUGMENT_NUM_FREQ_MASKS,
    num_time_masks: int = SPEC_AUGMENT_NUM_TIME_MASKS,
) -> np.ndarray:
    """
    Apply SpecAugment-style frequency and time masking to a log-Mel
    spectrogram, IN-PLACE-SAFE (operates on and returns a copy).

    This is a direct, minimal adaptation of the masking step from the
    SpecAugment paper (Park et al., 2019), applied here to the Mel
    spectrogram because it is the only genuinely 2-D (frequency x time)
    representation computed in this pipeline -- everything else (MFCC,
    chroma, spectral contrast, tonnetz) is mean-pooled immediately and has
    no spatial structure left to mask meaningfully. Masking happens BEFORE
    pooling, so it actually perturbs the information the model sees,
    rather than masking an already-collapsed scalar.

    Frequency masking: zeroes out ``num_freq_masks`` contiguous bands of
    mel bins (each up to ``freq_mask_param`` bins wide) across all time
    frames -- simulates losing a range of frequencies.

    Time masking: zeroes out ``num_time_masks`` contiguous blocks of time
    frames (each up to ``time_mask_param`` frames wide) across all mel
    bins -- simulates a brief dropout/occlusion in the recording.

    Parameters
    ----------
    mel_db : np.ndarray
        Log-scaled Mel spectrogram of shape ``(n_mels, n_time_frames)``,
        as produced by ``librosa.power_to_db(melspectrogram(...))``.
    freq_mask_param : int, default=12
        Maximum width (in mel bins) of each frequency mask.
    time_mask_param : int, default=8
        Maximum width (in time frames) of each time mask.
    num_freq_masks : int, default=1
        Number of frequency masks to apply.
    num_time_masks : int, default=1
        Number of time masks to apply.

    Returns
    -------
    np.ndarray
        A masked COPY of ``mel_db``, same shape and dtype. The input array
        is never modified.
    """
    masked = mel_db.copy()
    n_mels, n_frames = masked.shape

    # --- Frequency masking: zero out bands of mel bins ---
    for _ in range(num_freq_masks):
        if n_mels <= 1:
            break
        mask_width = np.random.randint(0, min(freq_mask_param, n_mels) + 1)
        if mask_width == 0:
            continue
        start = np.random.randint(0, n_mels - mask_width + 1)
        masked[start : start + mask_width, :] = masked.mean()

    # --- Time masking: zero out blocks of time frames ---
    for _ in range(num_time_masks):
        if n_frames <= 1:
            break
        mask_width = np.random.randint(0, min(time_mask_param, n_frames) + 1)
        if mask_width == 0:
            continue
        start = np.random.randint(0, n_frames - mask_width + 1)
        masked[:, start : start + mask_width] = masked.mean()

    return masked


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_features_from_array(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    apply_spec_augment: bool = False,
) -> Optional[np.ndarray]:
    """
    Extract the full mean-pooled acoustic feature vector from an in-memory
    waveform array (no file I/O). This is the shared core used by both
    ``extract_features`` (file-based) and the augmentation pipeline, so
    that augmented waveforms never need to be written to disk.

    Feature blocks (all mean-pooled across time, concatenated in order):

        1. MFCC                  (n_mfcc,)
        2. Delta-MFCC             (n_mfcc,)
        3. Delta-Delta-MFCC       (n_mfcc,)
        4. Chroma STFT            (12,)
        5. Mel Spectrogram (dB)   (128,)
        6. Spectral contrast      (7,)
        7. Tonnetz                (6,)
        8. Zero-crossing rate     (1,)
        9. RMS energy             (1,)

    Parameters
    ----------
    audio : np.ndarray
        1-D waveform array.
    sample_rate : int, default=22050
        Sample rate of ``audio``.
    n_mfcc : int, default=40
        Number of MFCC coefficients to compute.
    apply_spec_augment : bool, default=False
        If True, apply SpecAugment-style frequency/time masking (see
        ``spec_augment_mel``) to the Mel spectrogram BEFORE it is
        mean-pooled. This is intended ONLY for additional training-time
        augmentation; it must never be set True when extracting features
        for the held-out validation/test split or for inference
        (``predict.py``), since masking is a form of training-only noise
        injection, not a transform that should alter evaluation data.

    Returns
    -------
    Optional[np.ndarray]
        A 1-D array of shape ``(FEATURE_DIM,)``, or ``None`` if extraction
        failed for any reason.
    """
    if audio is None or len(audio) == 0:
        logger.warning("Empty waveform passed to extract_features_from_array; skipping.")
        return None

    try:
        # --- Timbre: MFCC + first/second derivatives ---
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)
        mfcc_delta2_mean = np.mean(mfcc_delta2.T, axis=0)

        # --- Pitch: chroma (energy per pitch class) ---
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)

        # --- Perceptual frequency energy: Mel spectrogram (log-scaled) ---
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if apply_spec_augment:
            mel_db = spec_augment_mel(mel_db)
        mel_mean = np.mean(mel_db.T, axis=0)

        # --- Spectral shape: contrast between spectral peaks and valleys ---
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        contrast_mean = np.mean(contrast.T, axis=0)

        # --- Harmonic/tonal centroid space (works best on harmonic signal) ---
        harmonic = librosa.effects.harmonic(audio)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sample_rate)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)

        # --- Energy / voicing descriptors ---
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr.T, axis=0)

        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms.T, axis=0)

        feature_vector = np.concatenate(
            [
                mfcc_mean,
                mfcc_delta_mean,
                mfcc_delta2_mean,
                chroma_mean,
                mel_mean,
                contrast_mean,
                tonnetz_mean,
                zcr_mean,
                rms_mean,
            ]
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to compute features from waveform: %s", exc)
        return None

    return feature_vector.astype(np.float32)


def extract_features(
    file_path: str,
    n_mfcc: int = N_MFCC,
    sample_rate: int = SAMPLE_RATE,
) -> Optional[np.ndarray]:
    """
    Extract the full mean-pooled acoustic feature vector from a single
    audio file on disk. Thin wrapper around ``extract_features_from_array``
    that handles file loading.

    Parameters
    ----------
    file_path : str
        Path to the ``.wav`` file.
    n_mfcc : int, default=40
        Number of MFCC coefficients to compute.
    sample_rate : int, default=22050
        Sample rate to resample the audio to during loading.

    Returns
    -------
    Optional[np.ndarray]
        A 1-D array of shape ``(FEATURE_DIM,)`` containing the concatenated
        pooled feature blocks, or ``None`` if the file could not be
        read/processed.
    """
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
    except Exception as exc:  # noqa: BLE001 - we want to catch any librosa/IO error
        logger.error("Failed to load audio file '%s': %s", file_path, exc)
        return None

    return extract_features_from_array(audio, sample_rate=sr, n_mfcc=n_mfcc)


def extract_mfcc(
    file_path: str,
    n_mfcc: int = N_MFCC,
    sample_rate: int = SAMPLE_RATE,
) -> Optional[np.ndarray]:
    """
    Backward-compatible alias for :func:`extract_features`.

    Retained under the original name so that any external code or notebooks
    written against the earlier API keep working. New code should call
    :func:`extract_features` directly.
    """
    return extract_features(file_path, n_mfcc=n_mfcc, sample_rate=sample_rate)


# --------------------------------------------------------------------------- #
# Dataset traversal
# --------------------------------------------------------------------------- #
def discover_dataset_files(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> Tuple[List[Path], List[int]]:
    """
    Walk every ``Actor_XX`` subdirectory under ``data_dir`` (at any nesting
    depth) and return the list of valid ``.wav`` file paths together with
    their parsed integer emotion labels. Performs NO audio loading or
    feature extraction -- this is purely a filesystem + filename-parsing
    step, intended to run BEFORE any train/test split decision is made.

    Parameters
    ----------
    data_dir : str or Path, default="data/ravdess"
        Root directory containing the ``Actor_01`` ... ``Actor_24``
        subdirectories (at any depth - found via recursive glob).

    Returns
    -------
    Tuple[List[Path], List[int]]
        Parallel lists: ``file_paths[i]`` has label ``labels[i]``. Files
        whose names don't match the RAVDESS convention are silently
        skipped (with a warning logged by ``parse_emotion_from_filename``).

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` does not exist.
    RuntimeError
        If no valid, parseable ``.wav`` files are found.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' does not exist. "
            f"Expected structure: {data_dir}/.../Actor_01/, Actor_02/, ..."
        )

    all_wav_paths = sorted(data_dir.rglob("*.wav"))
    logger.info("Found %d .wav files under '%s'.", len(all_wav_paths), data_dir)

    if not all_wav_paths:
        raise RuntimeError(
            f"No .wav files were found under '{data_dir}'. "
            "Please verify the dataset has been downloaded and extracted correctly."
        )

    file_paths: List[Path] = []
    labels: List[int] = []

    for wav_path in all_wav_paths:
        label = parse_emotion_from_filename(wav_path.name)
        if label is None:
            continue
        file_paths.append(wav_path)
        labels.append(label)

    if not file_paths:
        raise RuntimeError(
            "No files with parseable RAVDESS emotion codes were found. "
            "Check that filenames follow the '03-01-EE-...' convention."
        )

    logger.info(
        "Discovered %d valid labeled files (%d unparseable files skipped).",
        len(file_paths),
        len(all_wav_paths) - len(file_paths),
    )

    return file_paths, labels


def split_dataset_files(
    file_paths: List[Path],
    labels: List[int],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Path], List[Path], List[int], List[int]]:
    """
    Split the ORIGINAL (un-augmented) file list into train/test subsets
    using ``GroupShuffleSplit``, with each source file's own path as its
    group key.

    This is the leakage-prevention step: splitting happens here, on the
    1,440 raw RAVDESS files, BEFORE any augmentation or SpecAugment
    expansion exists. Every augmented row generated later for a given file
    inherits that file's train/test assignment, so it is structurally
    impossible for an augmented variant of a test-split utterance to end
    up in the training set (or vice versa) -- there is nothing left to
    "leak" across, since the grouping decision is made before expansion.

    ``GroupShuffleSplit`` (rather than plain ``train_test_split``) is used
    deliberately: at this stage every file is trivially its own group (one
    row each), so the numeric result is equivalent to a plain shuffled
    split, but using the group-aware splitter makes the
    "no group straddles both sides" guarantee structural and explicit
    rather than incidental. If this function is ever extended to operate
    on an already-expanded (post-augmentation) row list, the group-based
    guarantee continues to hold automatically.

    Parameters
    ----------
    file_paths : List[Path]
        Source file paths, as returned by ``discover_dataset_files``.
    labels : List[int]
        Parallel list of integer emotion labels.
    test_size : float, default=0.2
        Fraction of FILES (not augmented rows) to hold out for testing.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[List[Path], List[Path], List[int], List[int]]
        ``(train_file_paths, test_file_paths, train_labels, test_labels)``.

    Raises
    ------
    ValueError
        If ``file_paths`` and ``labels`` have different lengths, or if
        either resulting split would be empty.
    """
    if len(file_paths) != len(labels):
        raise ValueError(
            f"file_paths ({len(file_paths)}) and labels ({len(labels)}) "
            "must have the same length."
        )

    # Each file is its own group: the group key is simply the file's index
    # (equivalently, its path), since at this stage there is exactly one
    # row per file. This makes GroupShuffleSplit guarantee that no group
    # (and therefore no source file, and therefore none of its eventual
    # augmented children) is split across train and test.
    n_files = len(file_paths)
    groups = np.arange(n_files)
    labels_arr = np.array(labels)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X=groups, y=labels_arr, groups=groups))

    train_file_paths = [file_paths[i] for i in train_idx]
    test_file_paths = [file_paths[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]

    if not train_file_paths or not test_file_paths:
        raise ValueError(
            "GroupShuffleSplit produced an empty train or test split. "
            "Check test_size and the number of input files."
        )

    # Defensive verification: confirm zero overlap between the two file
    # sets. This should be mathematically guaranteed by GroupShuffleSplit,
    # but we check explicitly since this invariant is the entire point of
    # this function.
    train_set = {str(p) for p in train_file_paths}
    test_set = {str(p) for p in test_file_paths}
    overlap = train_set & test_set
    if overlap:
        raise RuntimeError(
            f"INTERNAL ERROR: {len(overlap)} file(s) appear in both the train "
            f"and test splits: {sorted(overlap)[:5]}... This should be "
            "impossible with GroupShuffleSplit and indicates a bug."
        )

    logger.info(
        "Split %d source files -> %d train / %d test (test_size=%.2f). "
        "Verified zero file overlap between splits.",
        n_files,
        len(train_file_paths),
        len(test_file_paths),
        test_size,
    )

    return train_file_paths, test_file_paths, train_labels, test_labels


def process_files(
    file_paths: List[Path],
    labels: List[int],
    n_mfcc: int = N_MFCC,
    sample_rate: int = SAMPLE_RATE,
    augment: bool = False,
    spec_augment: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature rows for an EXPLICIT, already-decided list of audio
    files (e.g. just the training split's files, or just the test split's
    files). This is the shared extraction engine used by ``process_dataset``
    for both the train and test sides of the split, with different
    ``augment``/``spec_augment`` settings per call.

    Parameters
    ----------
    file_paths : List[Path]
        Audio file paths to process.
    labels : List[int]
        Parallel list of integer emotion labels, one per file in
        ``file_paths``.
    n_mfcc : int, default=40
        Number of MFCC coefficients to extract per file.
    sample_rate : int, default=22050
        Sample rate used for audio loading.
    augment : bool, default=False
        If True, generate and extract features for augmented copies
        (noise, pitch shift, time stretch) of every waveform IN ADDITION
        TO the original, multiplying the row count for this file list.
        Pass ``True`` only for the training file list -- validation/test
        files must be processed with ``augment=False`` so the held-out
        split contains only clean, original audio.
    spec_augment : bool, default=False
        If True, additionally apply SpecAugment-style frequency/time
        masking to the Mel-spectrogram of every augmented variant (never
        the original waveform's row, and only relevant when
        ``augment=True``). Pass ``True`` only for the training file list.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``X`` of shape ``(n_rows, FEATURE_DIM)`` and ``y`` of shape
        ``(n_rows,)``.

    Raises
    ------
    ValueError
        If ``file_paths`` and ``labels`` have different lengths.
    RuntimeError
        If no valid samples could be extracted.
    """
    if len(file_paths) != len(labels):
        raise ValueError(
            f"file_paths ({len(file_paths)}) and labels ({len(labels)}) "
            "must have the same length."
        )

    features: List[np.ndarray] = []
    out_labels: List[int] = []
    n_failed = 0

    for idx, (wav_path, label) in enumerate(zip(file_paths, labels), start=1):
        try:
            audio, sr = librosa.load(str(wav_path), sr=sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load audio file '%s': %s", wav_path, exc)
            n_failed += 1
            continue

        # --- Original waveform: NEVER spec-augmented, regardless of split ---
        feature_vector = extract_features_from_array(
            audio, sample_rate=sr, n_mfcc=n_mfcc, apply_spec_augment=False
        )
        if feature_vector is None:
            n_failed += 1
            continue

        features.append(feature_vector)
        out_labels.append(label)

        # --- Augmented waveforms: only generated when augment=True ---
        # (the caller is responsible for only passing augment=True for the
        # TRAINING file list; this function applies no judgment of its own
        # about which split it's being used for).
        if augment:
            for aug_audio in generate_augmented_waveforms(audio, sample_rate=sr):
                aug_feature_vector = extract_features_from_array(
                    aug_audio,
                    sample_rate=sr,
                    n_mfcc=n_mfcc,
                    apply_spec_augment=spec_augment,
                )
                if aug_feature_vector is None:
                    n_failed += 1
                    continue
                features.append(aug_feature_vector)
                out_labels.append(label)

        if idx % 100 == 0 or idx == len(file_paths):
            logger.info(
                "Processed %d/%d files in this split (rows so far: %d)...",
                idx,
                len(file_paths),
                len(features),
            )

    if not features:
        raise RuntimeError(
            "Feature extraction produced zero valid samples for this file list. "
            "Check that the dataset files are not corrupted."
        )

    X = np.stack(features).astype(np.float32)
    y = np.array(out_labels, dtype=np.int64)

    logger.info(
        "Extraction complete for this split. X.shape=%s, y.shape=%s, failed=%d, "
        "augment=%s, spec_augment=%s",
        X.shape,
        y.shape,
        n_failed,
        augment,
        spec_augment,
    )

    return X, y


def process_dataset(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    n_mfcc: int = N_MFCC,
    sample_rate: int = SAMPLE_RATE,
    augment: bool = True,
    spec_augment: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Leakage-safe end-to-end dataset preparation:

        1. Discover all original RAVDESS files (``discover_dataset_files``).
        2. Split them into train/test FILE LISTS using ``GroupShuffleSplit``
           on source-file identity (``split_dataset_files``) -- this
           happens BEFORE any augmentation exists.
        3. Extract features for the training files WITH augmentation and
           (optionally) SpecAugment (``process_files(..., augment=True)``).
        4. Extract features for the test files WITHOUT any augmentation or
           masking (``process_files(..., augment=False)``) -- the held-out
           split contains only clean, original audio.

    Because the split decision is made on the 1,440 original files before
    any augmented rows are generated, it is structurally impossible for an
    augmented variant of a test-split utterance to appear in the training
    data, or vice versa.

    Parameters
    ----------
    data_dir : str or Path, default="data/ravdess"
        Root directory containing the ``Actor_01`` ... ``Actor_24``
        subdirectories (at any depth - found via recursive glob).
    n_mfcc : int, default=40
        Number of MFCC coefficients to extract per file.
    sample_rate : int, default=22050
        Sample rate used for audio loading.
    augment : bool, default=True
        If True, generate augmented copies (noise, pitch shift, time
        stretch) for TRAINING files only.
    spec_augment : bool, default=True
        If True, additionally apply SpecAugment-style masking to augmented
        TRAINING rows only. Has no effect if ``augment=False``.
    test_size : float, default=0.2
        Fraction of ORIGINAL FILES (not augmented rows) to hold out for
        the test split.
    random_state : int, default=42
        Random seed for the file-level split, for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(X_train, X_test, y_train, y_test)``. ``X_train``/``y_train``
        include augmented rows when ``augment=True``; ``X_test``/``y_test``
        contain ONLY original, unaugmented, unmasked audio.

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` does not exist.
    RuntimeError
        If no valid samples could be extracted, or if the internal
        train/test file-overlap check fails.
    """
    file_paths, labels = discover_dataset_files(data_dir=data_dir)

    train_paths, test_paths, train_labels, test_labels = split_dataset_files(
        file_paths, labels, test_size=test_size, random_state=random_state
    )

    logger.info("Extracting features for the TRAINING split (augment=%s)...", augment)
    X_train, y_train = process_files(
        train_paths,
        train_labels,
        n_mfcc=n_mfcc,
        sample_rate=sample_rate,
        augment=augment,
        spec_augment=spec_augment,
    )

    logger.info("Extracting features for the TEST split (augment=False, always clean)...")
    X_test, y_test = process_files(
        test_paths,
        test_labels,
        n_mfcc=n_mfcc,
        sample_rate=sample_rate,
        augment=False,
        spec_augment=False,
    )

    logger.info(
        "Dataset preparation complete. X_train=%s, y_train=%s, X_test=%s, y_test=%s",
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
    )

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
def save_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    x_train_path: str | Path = DEFAULT_X_TRAIN_PATH,
    x_test_path: str | Path = DEFAULT_X_TEST_PATH,
    y_train_path: str | Path = DEFAULT_Y_TRAIN_PATH,
    y_test_path: str | Path = DEFAULT_Y_TEST_PATH,
) -> None:
    """
    Save the pre-split train/test feature matrices and label vectors to
    disk as ``.npy`` files.

    Saving the split explicitly (rather than one combined ``X``/``y`` pair
    that gets re-split later) is the persistence-layer half of the
    leakage fix: once train and test are written to separate files here,
    no downstream code can accidentally re-shuffle or re-split rows across
    the boundary that was carefully constructed in ``process_dataset``.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices of shape ``(n_train_samples, FEATURE_DIM)`` and
        ``(n_test_samples, FEATURE_DIM)``.
    y_train, y_test : np.ndarray
        Integer label vectors, parallel to ``X_train``/``X_test``.
    x_train_path, x_test_path, y_train_path, y_test_path : str or Path
        Output paths for each array. Defaults write into
        ``data/processed/``.
    """
    paths = [x_train_path, x_test_path, y_train_path, y_test_path]
    arrays = [X_train, X_test, y_train, y_test]

    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    for path, array in zip(paths, arrays):
        np.save(path, array)

    logger.info(
        "Saved split features: X_train=%s -> '%s', X_test=%s -> '%s', "
        "y_train=%s -> '%s', y_test=%s -> '%s'.",
        X_train.shape,
        x_train_path,
        X_test.shape,
        x_test_path,
        y_train.shape,
        y_train_path,
        y_test.shape,
        y_test_path,
    )


def load_features(
    x_train_path: str | Path = DEFAULT_X_TRAIN_PATH,
    x_test_path: str | Path = DEFAULT_X_TEST_PATH,
    y_train_path: str | Path = DEFAULT_Y_TRAIN_PATH,
    y_test_path: str | Path = DEFAULT_Y_TEST_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load previously saved, pre-split train/test feature matrices and label
    vectors from disk.

    Parameters
    ----------
    x_train_path, x_test_path, y_train_path, y_test_path : str or Path
        Paths to the saved arrays. Defaults read from ``data/processed/``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(X_train, X_test, y_train, y_test)``.

    Raises
    ------
    FileNotFoundError
        If any of the four files does not exist.
    """
    paths = [x_train_path, x_test_path, y_train_path, y_test_path]
    missing = [str(p) for p in paths if not Path(p).exists()]

    if missing:
        raise FileNotFoundError(
            f"Processed feature file(s) not found: {missing}. "
            "Run process_dataset() + save_features() first."
        )

    X_train, X_test, y_train, y_test = (np.load(p) for p in paths)

    logger.info(
        "Loaded split features: X_train=%s, X_test=%s, y_train=%s, y_test=%s.",
        X_train.shape,
        X_test.shape,
        y_train.shape,
        y_test.shape,
    )

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Script entry point (allows running this file standalone)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("Starting standalone feature extraction run (leakage-safe split)...")
    X_train_data, X_test_data, y_train_data, y_test_data = process_dataset(
        augment=True, spec_augment=False
    )
    save_features(X_train_data, X_test_data, y_train_data, y_test_data)