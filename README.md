# 🎙️ Emotion Recognition from Speech

A deep learning project developed during the CodeAlpha Machine Learning Internship that classifies human emotions from speech audio using acoustic features and a CNN-BiLSTM network.

---

## 📌 Overview

The system analyzes speech recordings from the RAVDESS dataset and predicts one of eight emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## Dataset

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

- 24 actors
- 1440 speech recordings
- 8 emotion classes

---

## Feature Extraction

The following audio features are extracted using Librosa:

- MFCC
- Delta MFCC
- Delta-Delta MFCC
- Chroma Features
- Mel Spectrogram
- Spectral Contrast
- Tonnetz
- Zero Crossing Rate
- RMS Energy

Total Feature Dimension: **275**

---

## Model Architecture

CNN + Bidirectional LSTM

```
Input (275×1)

↓ Conv1D(64)
↓ Batch Normalization
↓ Max Pooling
↓ Dropout

↓ Conv1D(128)
↓ Batch Normalization
↓ Max Pooling
↓ Dropout

↓ Bidirectional LSTM(128)

↓ Dense(256)
↓ Dropout

↓ Dense(128)
↓ Dropout

↓ Softmax (8 classes)
```

---

## Training Techniques

- StandardScaler
- Class Weights
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- Data Augmentation

---

## Project Structure

```
CodeAlpha_EmotionRecognitionFromSpeech/
│
├── data/
├── models/
├── plots/
├── notebooks/
├── src/
│   ├── feature_extraction.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_EmotionRecognitionFromSpeech.git

cd CodeAlpha_EmotionRecognitionFromSpeech

pip install -r requirements.txt
```

---

## Run

Train and evaluate the model:

```bash
python main.py
```

Force feature extraction:

```bash
python main.py --force-extract
```

---

## Libraries Used

- TensorFlow
- NumPy
- Librosa
- Scikit-learn
- Matplotlib
- Joblib

---

## Future Improvements

- Attention Mechanisms
- Transformer-based Models
- Hyperparameter Optimization
- Real-time Emotion Recognition
- Streamlit Web Application

---

## Author

**Vignesh R**

Artificial Intelligence & Machine Learning Student

CodeAlpha Machine Learning Internship
