"""
model.py

Defines the CNN architecture used to classify chest X-rays into
COVID / NonCOVID / Normal categories.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape=(128, 128, 1), num_classes=3):
    """
    A compact CNN: three convolutional blocks (conv -> batchnorm -> pool),
    followed by a dense classification head with dropout for regularization.
    Kept intentionally small so it trains quickly on CPU and on small datasets,
    while still being a genuine learnable architecture (not a toy stub).
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_cnn()
    model.summary()
