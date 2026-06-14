# Handwritten Digit Recognition

A complete deep-learning system for recognising handwritten digits (0-9), built for
Educational Practice 2025-2026 (Deep Learning track).

A convolutional neural network (CNN) is trained on the MNIST dataset and reaches
**99.51% accuracy** on the 10,000-image test set. A Flask web application lets you
draw a digit with your mouse or finger and see the model's live prediction.

## Project structure

```
.
├── src/
│   ├── data_loader.py   # Loads, normalises, and reshapes MNIST data
│   ├── model.py          # CNN architecture definition
│   ├── train.py           # Training script (saves model + history plot)
│   └── evaluate.py        # Evaluation: accuracy, classification report, plots
├── app/
│   ├── app.py             # Flask server
│   ├── preprocessing.py   # Canvas-image -> MNIST-format preprocessing
│   └── templates/
│       └── index.html     # Drawing canvas + prediction UI
├── models/                 # Saved trained model (digit_cnn.keras)
├── results/                # Training history, confusion matrix, sample predictions
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the model

```bash
cd src
python train.py
```

This downloads MNIST (falling back to a public mirror if needed), trains the CNN
for up to 20 epochs (early stopping usually finishes around epoch 8), and saves:

- `models/digit_cnn.keras` — the trained model
- `results/training_history.png` — accuracy/loss curves

## 2. Evaluate the model

```bash
cd src
python evaluate.py
```

Prints overall accuracy and a per-class precision/recall/F1 report, and saves:

- `results/confusion_matrix.png`
- `results/sample_predictions.png`

## 3. Run the web app

```bash
cd app
python app.py
```

Open `http://localhost:5000` in your browser, draw a digit, and click
**Read digit** (prediction also runs automatically when you finish a stroke).

## Model architecture

A compact VGG-style CNN (272,106 parameters):

- **Block 1**: Conv(32) → BatchNorm → Conv(32) → MaxPool → Dropout(0.25)
- **Block 2**: Conv(64) → BatchNorm → Conv(64) → MaxPool → Dropout(0.25)
- **Head**: Flatten → Dense(128) → BatchNorm → Dropout(0.5) → Dense(10, softmax)

Trained with the Adam optimiser and sparse categorical cross-entropy loss, with
early stopping and learning-rate reduction on plateau.

## Results

| Metric | Value |
|---|---|
| Test accuracy | 99.51% |
| Test loss | low, validation tracks training closely (no over-fitting) |

Every digit class scores above 0.99 F1. See `results/` for plots after running
`train.py` and `evaluate.py`.

## Team

See the project report (`report.pdf`) for team members and contributions.
