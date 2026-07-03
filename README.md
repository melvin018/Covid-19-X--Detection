# COVID-19 X-Ray and Lung Mask Classification

A convolutional neural network pipeline for classifying chest X-ray images into **COVID**, **NonCOVID**, and **Normal** categories, built with TensorFlow/Keras, OpenCV, and scikit-learn.

## Pipeline Overview

```
generate_sample_data.py  →  preprocess.py  →  train.py  →  evaluate.py
   (sample data)             (load/clean)      (train CNN)   (report + confusion matrix)
```

1. **`generate_sample_data.py`** — Generates a small synthetic sample dataset matching the expected folder structure, for testing the pipeline without a real dataset on hand.
2. **`preprocess.py`** — Loads images, converts to grayscale, resizes, and normalizes pixel values using OpenCV.
3. **`model.py`** — Defines a 3-block CNN (Conv2D → BatchNorm → MaxPool, repeated) with a dense classification head.
4. **`train.py`** — Splits data into train/validation sets, applies light geometric augmentation, trains with early stopping, and saves the model plus a loss/accuracy curve plot.
5. **`evaluate.py`** — Loads the trained model, generates a classification report (precision/recall/F1 per class), and saves a confusion matrix plot.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate sample data (skip this step if you have a real dataset)
python generate_sample_data.py

# 2. Verify preprocessing
python preprocess.py

# 3. Train the model
python train.py

# 4. Evaluate on the validation split
python evaluate.py
```

## Using a Real Dataset

Replace the contents of `data/raw/<ClassName>/` with real X-ray images (`.png`/`.jpg`), keeping the same folder structure:

```
data/raw/COVID/*.png
data/raw/NonCOVID/*.png
data/raw/Normal/*.png
```

A commonly used public dataset with this exact structure is the **COVID-19 Radiography Database** (Tawsifur Rahman et al., available on Kaggle). No code changes are required — `preprocess.py`, `train.py`, and `evaluate.py` are dataset-agnostic.

## Notes on the Sample Data

This repository ships with a synthetic sample dataset generator (`generate_sample_data.py`) so the full pipeline can be run and verified without requiring a large medical imaging dataset upfront. The synthetic images are **not real X-rays** and the accuracy numbers produced when training on them are **not meaningful as real-world performance** — they exist to confirm the pipeline (data loading, preprocessing, training, evaluation) runs correctly end-to-end. Swap in real data for meaningful results.

## Tech Stack

Python, TensorFlow/Keras, OpenCV, Matplotlib, scikit-learn, NumPy
