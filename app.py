"""
Flask web application for the Handwritten Digit Recognition project.

Loads the trained CNN once at start-up. When the browser sends a drawing,
the server preprocesses it to match the MNIST format, runs the model, and
returns the predicted digit, confidence, the full probability distribution,
and a base64-encoded preview of the 28x28 image the model actually sees.
"""

import os
import re
import base64
import io

import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow import keras

from preprocessing import preprocess_canvas_image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "digit_cnn.keras")

app = Flask(__name__)

print(f"Loading model from {MODEL_PATH}...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # The canvas sends a data URL like "data:image/png;base64,...."
    image_data = data["image"]
    match = re.match(r"^data:image/\w+;base64,(.*)$", image_data)
    image_b64 = match.group(1) if match else image_data

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"error": "Invalid image data"}), 400

    model_input, preview = preprocess_canvas_image(image_bytes)

    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    # Encode the 28x28 preview as a base64 PNG for the front end
    buffer = io.BytesIO()
    preview.resize((140, 140), resample=0).save(buffer, format="PNG")
    preview_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({
        "prediction": predicted_digit,
        "confidence": confidence,
        "probabilities": [float(p) for p in probabilities],
        "preview": f"data:image/png;base64,{preview_b64}",
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
