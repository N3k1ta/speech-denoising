# Speech Denoising with Deep Learning

Speech denoising system trained on the VoiceBank+DEMAND dataset.
Baseline project toward a real-time VST/AU plugin based on DeepFilterNet + JUCE.

## Dataset
- **VoiceBank** (clean speech) + **DEMAND** (real-world noise environments)
- 11,572 training pairs / 824 test pairs
- Sample rate: 16,000 Hz

## Notebooks

| Notebook | Description |
|----------|-------------|
| `speech_denoising_01.ipynb` | EDA & Preprocessing |
| `speech_denoising_02.ipynb` | Baseline 2-layer CNN, MSELoss, 3 epochs |
| `speech_denoising_03.ipynb` | 4-layer CNN + validation loop |
| `speech_denoising_04.ipynb` | Dilated CNN — wider receptive field |
| `speech_denoising_05.ipynb` | Spectral loss (STFT magnitude + Hann window) |
| `speech_denoising_06.ipynb` | Evaluation — PESQ & STOI vs noisy baseline |

## Results

| | PESQ | STOI |
|---|---|---|
| Noisy input (no model) | 2.10 | 0.90 |
| Model output | 2.04 | 0.87 |

Shallow dilated CNN baseline, 4 epochs. Informal listening tests show perceptible noise reduction despite lower objective scores.

## Project Structure
```
data/
  train_clean_chunks.npy
  train_noisy_chunks.npy
  test_clean_chunks.npy
  test_noisy_chunks.npy
speech_denoising_01.ipynb
speech_denoising_02.ipynb
speech_denoising_03.ipynb
speech_denoising_04.ipynb
speech_denoising_05.ipynb
speech_denoising_06.ipynb
```

## Stack
Python · PyTorch · librosa · NumPy · pesq · pystoi