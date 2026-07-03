"""
train.py

Trains the CNN on the preprocessed X-ray dataset, with:
  - train/validation split
  - light data augmentation (rotation, flip, zoom) via Keras layers
  - early stopping to avoid overfitting on the small sample set
  - saved model weights + training history plot (Matplotlib)

Run after preprocess.py / generate_sample_data.py have produced data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from preprocess import load_dataset, IMG_SIZE
from model import build_cnn

OUT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(OUT_DIR, "covid_cnn_model.keras")
HISTORY_PLOT_PATH = os.path.join(OUT_DIR, "training_history.png")

EPOCHS = 25
BATCH_SIZE = 8
SEED = 42


def build_augmentation():
    """Light augmentation appropriate for medical-style imagery: no heavy
    color jitter (grayscale already), modest geometric variation only."""
    return keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
    ])


def main():
    print("Loading dataset...")
    X, y, class_names = load_dataset()
    print(f"Loaded {X.shape[0]} images across classes: {class_names}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Validation: {X_val.shape[0]}")

    augment = build_augmentation()
    model = build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=len(class_names))

    # Wrap augmentation as a preprocessing layer applied only at train time
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = augment(inputs)
    outputs = model(x)
    train_model = keras.Model(inputs, outputs)
    train_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )

    print("\nStarting training...\n")
    history = train_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    # Save the underlying (non-augmentation-wrapped) model for inference
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train acc")
    axes[1].plot(history.history["val_accuracy"], label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=120)
    print(f"Training history plot saved to {HISTORY_PLOT_PATH}")

    final_val_acc = history.history["val_accuracy"][-1]
    print(f"\nFinal validation accuracy: {final_val_acc:.2%}")
    print(
        "\nNote: this run used a small synthetic sample dataset to validate "
        "the pipeline. Accuracy numbers are not meaningful as a real-world "
        "performance claim until trained on the real COVID-19 X-ray dataset."
    )


if __name__ == "__main__":
    main()
