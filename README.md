# Speech Denoising with Deep Learning

Speech denoising system trained on the VoiceBank+DEMAND dataset.

## Dataset
- **VoiceBank** (clean speech) + **DEMAND** (real-world noise environments)
- 11,572 training pairs / 824 test pairs
- Sample rate: 16,000 Hz

## Notebook 01 — EDA & Preprocessing
- Validated noisy/clean file pairing by count, duration, and listening test
- Visualized STFT spectrograms (clean vs noisy)
- Sliced all audio into 1-second chunks (16,000 samples)
- Saved preprocessed arrays to `.npy` for fast loading during training

## Project Structure
data/
train_clean_chunks.npy
train_noisy_chunks.npy
test_clean_chunks.npy
test_noisy_chunks.npy
speech_denoising_01.ipynb  # EDA & Preprocessing
speech_denoising_02.ipynb  # Model Architecture & Training

## Stack
Python · PyTorch · librosa · NumPy
