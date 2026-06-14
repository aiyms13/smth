"""
Preprocessing for canvas drawings sent from the browser.

Reproduces the steps used to build the original MNIST images so that a
freehand drawing matches the format the model was trained on:
1. Convert to grayscale.
2. Invert colours so strokes are bright (white) on a dark (black)
   background, like MNIST.
3. Crop to the digit's bounding box.
4. Scale the digit to fit inside a 20x20 box, preserving aspect ratio.
5. Paste the scaled digit onto a 28x28 black canvas.
6. Shift the image so the digit's centre of mass is at the centre of the
   28x28 frame.
"""

import io
import numpy as np
from PIL import Image
from scipy import ndimage


def _shift_to_center_of_mass(image):
    """Shift image content so its centre of mass is at the image centre."""
    cy, cx = ndimage.center_of_mass(image)
    rows, cols = image.shape
    shift_y = np.round(rows / 2.0 - cy).astype(int)
    shift_x = np.round(cols / 2.0 - cx).astype(int)
    return ndimage.shift(image, (shift_y, shift_x), cval=0)


def preprocess_canvas_image(image_bytes):
    """
    Convert raw image bytes (PNG from the canvas) into a (1, 28, 28, 1)
    float32 array ready to feed into the model.

    Returns:
        model_input: numpy array of shape (1, 28, 28, 1), values in [0, 1]
        preview_image: PIL.Image of the final 28x28 image (for the UI preview)
    """
    # Load image and convert to grayscale
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(image).astype("float32")

    # Invert so strokes are bright on a dark background, like MNIST
    arr = 255.0 - arr

    # Threshold out faint anti-aliasing noise
    arr[arr < 20] = 0

    # If the canvas is empty, return a blank 28x28 image
    if arr.max() == 0:
        blank = np.zeros((28, 28), dtype="float32")
        preview = Image.fromarray(blank.astype("uint8"))
        return blank.reshape(1, 28, 28, 1) / 255.0, preview

    # Crop to the digit's bounding box
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = arr[rmin:rmax + 1, cmin:cmax + 1]

    # Scale so the longer side fits in a 20-pixel box, preserving aspect ratio
    h, w = cropped.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))
    else:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))

    cropped_img = Image.fromarray(cropped.astype("uint8")).resize((new_w, new_h), Image.LANCZOS)
    resized = np.array(cropped_img).astype("float32")

    # Paste the resized digit onto a 28x28 black canvas, centred
    canvas = np.zeros((28, 28), dtype="float32")
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    # Shift so the centre of mass sits at the centre of the frame
    canvas = _shift_to_center_of_mass(canvas)
    canvas = np.clip(canvas, 0, 255)

    preview = Image.fromarray(canvas.astype("uint8"))

    model_input = (canvas / 255.0).reshape(1, 28, 28, 1).astype("float32")
    return model_input, preview
