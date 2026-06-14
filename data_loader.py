"""
Data loading utilities for the Handwritten Digit Recognition project.

Loads the MNIST dataset, normalises pixel values to [0, 1], and reshapes
images to (28, 28, 1) as expected by the CNN. If the standard Keras
download is unavailable (e.g. on a restricted network), it automatically
falls back to a public mirror.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Public mirror used as a fallback if the default MNIST source is unreachable.
MNIST_MIRROR_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def load_mnist_data():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        (x_train, y_train), (x_test, y_test):
            x_train: float32 array of shape (60000, 28, 28, 1), values in [0, 1]
            y_train: int array of shape (60000,), digit labels 0-9
            x_test:  float32 array of shape (10000, 28, 28, 1), values in [0, 1]
            y_test:  int array of shape (10000,), digit labels 0-9
    """
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    except Exception as e:
        print(f"Default MNIST source failed ({e}); falling back to public mirror...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(
            path=MNIST_MIRROR_URL
        )

    # Normalise pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add a single channel dimension -> (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape:  {x_test.shape}, y_test shape:  {y_test.shape}")
    print(f"Pixel value range: [{x_train.min()}, {x_train.max()}]")
