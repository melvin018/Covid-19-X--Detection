# 🫁 Medical Imaging AI — COVID-19 Chest X-Ray Classifier

> End-to-end CNN pipeline for chest X-ray classification: COVID / Non-COVID / Normal.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

---

## What it does

A complete medical imaging AI pipeline that classifies chest X-ray images into three categories — COVID, Non-COVID, and Normal — built for compatibility with the public [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).

---

## Architecture

```
Chest X-Ray Image
    ↓
Preprocessing Pipeline
(OpenCV grayscale → resize → normalize)
    ↓
3-Block CNN
Conv2D → BatchNorm → MaxPool → Dropout (x3)
    ↓
Dense Classification Head
    ↓
Class: COVID / Non-COVID / Normal
    ↓
Evaluation Report
(Per-class Precision / Recall / F1 + Confusion Matrix)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | TensorFlow / Keras |
| Image Processing | OpenCV |
| Evaluation | scikit-learn |
| Visualization | Matplotlib |

---

## Key Features

- **Dataset-agnostic pipeline** — preprocessing and training require zero code changes to swap in a real dataset
- **CPU-optimized** — architecture designed to train efficiently without GPU
- **Clinical-grade evaluation** — per-class precision, recall, F1 score, and confusion matrix reports generated automatically after training
- **Standardized preprocessing** — grayscale conversion, resizing, and normalization consistent with medical imaging standards

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/melvin018/Covid-19-X--Detection.git
cd Covid-19-X--Detection

# Install dependencies
pip install -r requirements.txt

# Generate sample data to validate the pipeline
python generate_sample_data.py

# Train the model
python train.py

# Evaluate
python evaluate.py
```

---

## Using Real Data

Download the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle and update the dataset path in `config.py`. The pipeline requires no other changes.

---

## Note on Results

This pipeline was validated on synthetic placeholder data to confirm the architecture and preprocessing work correctly. Real-world accuracy figures pending a training run on the full radiography dataset.

---

## Built by

**Melvin Raju** — [LinkedIn](https://www.linkedin.com/in/melvin-raju-0810451b9) · [Portfolio](https://melvin018.github.io)
