"""
Model definition for the Handwritten Digit Recognition project.

Defines a compact VGG-style CNN with two convolutional blocks followed by
a dense classifier head. Keeping the architecture in one place ensures the
training script, evaluation script, and web app all use the same definition.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Build and compile the CNN used for digit classification.

    Args:
        input_shape: shape of a single input image, default (28, 28, 1)
        num_classes: number of output classes, default 10 (digits 0-9)

    Returns:
        A compiled tf.keras.Model ready for training.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),

            # Convolutional block 1
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Convolutional block 2
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Classifier head
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="digit_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
