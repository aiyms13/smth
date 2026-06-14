"""
Evaluation script for the Handwritten Digit Recognition project.

Loads the saved model, runs it on the full MNIST test set, and reports:
- overall accuracy
- a per-class precision/recall/F1 table
- a confusion matrix heatmap
- a grid of example correct and incorrect predictions
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from data_loader import load_mnist_data

MODEL_PATH = "../models/digit_cnn.keras"
RESULTS_DIR = "../results"
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
SAMPLE_PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "sample_predictions.png")


def plot_confusion_matrix(cm, save_path):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title("Confusion matrix (test set)")
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    # Annotate non-zero cells with their counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value > 0:
                color = "white" if value > thresh else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved confusion matrix to {save_path}")


def plot_sample_predictions(x_test, y_true, y_pred, probabilities, save_path,
                              n_correct=10, n_incorrect=10):
    """Plot a grid of correct (top rows) and incorrect (bottom rows) predictions."""
    correct_idx = np.where(y_pred == y_true)[0]
    incorrect_idx = np.where(y_pred != y_true)[0]

    np.random.seed(42)
    correct_sample = np.random.choice(correct_idx, size=min(n_correct, len(correct_idx)), replace=False)
    incorrect_sample = np.random.choice(incorrect_idx, size=min(n_incorrect, len(incorrect_idx)), replace=False)

    all_idx = np.concatenate([correct_sample, incorrect_sample])
    n_cols = 5
    n_rows = int(np.ceil(len(all_idx) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for ax, idx in zip(axes, all_idx):
        image = x_test[idx].squeeze()
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        confidence = probabilities[idx][pred_label]

        ax.imshow(image, cmap="gray")
        color = "green" if pred_label == true_label else "red"
        ax.set_title(f"pred {pred_label} ({confidence*100:.0f}%)\ntrue {true_label}",
                      color=color, fontsize=9)
        ax.axis("off")

    # Hide any unused subplots
    for ax in axes[len(all_idx):]:
        ax.axis("off")

    fig.suptitle("Sample predictions  -  top: correct,  bottom: misclassified")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved sample predictions to {save_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading test data...")
    (_, _), (x_test, y_test) = load_mnist_data()

    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)

    print("Running predictions on the test set...")
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = (y_pred == y_test).mean()
    print(f"\nOverall test accuracy: {accuracy:.4f} "
          f"({(y_pred == y_test).sum()} of {len(y_test)} images classified correctly)\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, CONFUSION_MATRIX_PATH)

    plot_sample_predictions(x_test, y_test, y_pred, probabilities, SAMPLE_PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
