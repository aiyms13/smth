"""
Training script for the Handwritten Digit Recognition project.

Loads the data, holds out a validation split, builds the model, and trains
it with early stopping and learning-rate reduction callbacks. Saves the
trained model and a plot of the training history.
"""

import os
import matplotlib
matplotlib.use("Agg")  # allow running without a display (e.g. on a server)
import matplotlib.pyplot as plt

from tensorflow import keras

from data_loader import load_mnist_data
from model import build_model

MODEL_DIR = "../models"
RESULTS_DIR = "../results"
MODEL_PATH = os.path.join(MODEL_DIR, "digit_cnn.keras")
HISTORY_PLOT_PATH = os.path.join(RESULTS_DIR, "training_history.png")

EPOCHS = 20
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1


def plot_training_history(history, save_path):
    """Plot training/validation accuracy and loss curves and save to disk."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["accuracy"], marker="o", label="Train")
    axes[0].plot(history.history["val_accuracy"], marker="s", label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], marker="o", label="Train")
    axes[1].plot(history.history["val_loss"], marker="s", label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Training history")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved training history plot to {save_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    print("Building model...")
    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    print("Training model...")
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=2,
    )

    print(f"Saving trained model to {MODEL_PATH}")
    model.save(MODEL_PATH)

    plot_training_history(history, HISTORY_PLOT_PATH)

    # Quick sanity check on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
