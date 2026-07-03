"""
evaluate.py

Loads the trained model and produces:
  - a classification report (precision, recall, F1 per class)
  - a confusion matrix plot (Matplotlib)

Run after train.py has produced covid_cnn_model.keras.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from preprocess import load_dataset

OUT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(OUT_DIR, "covid_cnn_model.keras")
CONFUSION_MATRIX_PATH = os.path.join(OUT_DIR, "confusion_matrix.png")
SEED = 42


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Confusion matrix saved to {save_path}")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run train.py first."
        )

    print("Loading dataset and trained model...")
    X, y, class_names = load_dataset()
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    model = keras.models.load_model(MODEL_PATH)
    probs = model.predict(X_val, verbose=0)
    preds = np.argmax(probs, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_val, preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_val, preds)
    plot_confusion_matrix(cm, class_names, CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()
