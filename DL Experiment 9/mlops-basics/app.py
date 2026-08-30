import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig-exp8"

from pathlib import Path
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Resolve model path relative to this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "baseline_ann.keras"

print(f"Loading baseline ANN model from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")


def preprocess_image(image_file):
    """
    Preprocesses an incoming image file for the baseline ANN:
    1. Opens with PIL
    2. Converts to grayscale ('L')
    3. Resizes to (28, 28)
    4. Normalizes pixel values to [0, 1] (/ 255.0)
    5. Reshapes to (1, 784)
    """
    image = Image.open(image_file).convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 784)
    return img_array


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint for digit prediction.
    Accepts multipart file upload under 'file' key.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    try:
        processed_img = preprocess_image(file)
        predictions = model.predict(processed_img, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(round(float(np.max(predictions[0])), 4))

        return jsonify({
            "predicted_digit": predicted_digit,
            "confidence": confidence
        }), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    # Development mode:
    # app.run(debug=True, port=port)
    #
    # Production / containerized / CI mode (listen on all network interfaces):
    app.run(host="0.0.0.0", port=port)
