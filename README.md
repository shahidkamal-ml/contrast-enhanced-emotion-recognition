# Contrast-Enhanced Emotion Recognition

Deep learning project for facial emotion recognition on the [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset. The notebooks compare standard grayscale preprocessing against **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and evaluate multiple model architectures.

## Overview

Facial expression images are often low-contrast and unevenly lit. This project applies CLAHE via OpenCV to improve local contrast before training, then compares results across a custom CNN and a fine-tuned ResNet-18.

**Emotion classes:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## Repository Contents

| File | Description |
|------|-------------|
| `baseline.ipynb` | Grayscale baseline (no CLAHE) with a custom CNN |
| `customcnn.ipynb` | CLAHE preprocessing with the same custom CNN |
| `resnet.ipynb` | CLAHE preprocessing with fine-tuned ResNet-18 |
| `dl project report .pdf` | Full project write-up |
| `project slides.pdf` | Presentation slides |

## Models

### Custom EmotionCNN

A lightweight 3-block CNN with batch normalization, dropout, and two fully connected layers, designed for 48×48 grayscale inputs.

### ResNet-18

Pretrained ResNet-18 adapted for 1-channel input and 7 output classes. Most layers are frozen; only `layer4` and the final classifier are fine-tuned.

## Preprocessing Pipeline

Shared steps across notebooks:

- Resize to 48×48
- Random horizontal flip and rotation (±10°)
- Gaussian blur augmentation (30% probability)
- Normalization to `[-1, 1]`

**Contrast enhancement (CLAHE notebooks):**

```python
cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
```

## Results (25 epochs, test set)

| Notebook | Preprocessing | Model | Test Accuracy |
|----------|---------------|-------|---------------|
| `baseline.ipynb` | Grayscale | Custom CNN | **58.19%** |
| `customcnn.ipynb` | CLAHE | Custom CNN | 56.78% |
| `resnet.ipynb` | CLAHE | ResNet-18 (partial fine-tune) | 41.81% |

> Results were obtained in Google Colab with the FER2013 train/test split. Your numbers may vary slightly depending on hardware and random seed.

## Getting Started

The notebooks are designed to run in **Google Colab**.

1. Open a notebook in Colab.
2. Upload your Kaggle API credentials (`kaggle.json`) when prompted.
3. The notebook downloads FER2013 automatically:

   ```bash
   kaggle datasets download -d msambare/fer2013
   ```

4. Run all cells to train and evaluate the model.

### Dependencies

- Python 3
- PyTorch & Torchvision
- OpenCV (`opencv-python`)
- Kaggle CLI
- Matplotlib, NumPy, PIL

Install locally with:

```bash
pip install torch torchvision opencv-python kaggle matplotlib numpy pillow
```

## Project Structure

```
.
├── baseline.ipynb          # Baseline (no CLAHE)
├── customcnn.ipynb         # CLAHE + custom CNN
├── resnet.ipynb            # CLAHE + ResNet-18
├── dl project report .pdf  # Written report
└── project slides.pdf      # Slides
```

## License

This repository is for academic and educational use. The FER2013 dataset is subject to its own terms on [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).
