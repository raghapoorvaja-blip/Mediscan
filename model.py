"""
model.py
--------
Defines the CNN (Convolutional Neural Network), loads the trained model from disk
and runs predictions.

The network is deliberately small and simple so that
  * it trains in minutes on an ordinary laptop CPU, and
  * Grad-CAM (see gradcam.py) can easily read from its last convolution layer.
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide noisy TensorFlow log messages

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from preprocessing import IMG_SIZE

# ----------------------------------------------------------------------------
# File locations (can be changed with the MEDISCAN_MODEL_DIR environment variable)
# ----------------------------------------------------------------------------
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "brain_mri_model.keras")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

# Names of two special layers that gradcam.py needs to find
LAST_CONV_LAYER = "last_conv"
LOGITS_LAYER = "logits"

DEFAULT_CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Friendly names shown in the interface (folder names differ between datasets)
_DISPLAY_NAMES = {
    "glioma": "Glioma Tumor",
    "glioma_tumor": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "meningioma_tumor": "Meningioma Tumor",
    "pituitary": "Pituitary Tumor",
    "pituitary_tumor": "Pituitary Tumor",
    "notumor": "No Tumor",
    "no_tumor": "No Tumor",
}


def display_name(class_name):
    """Convert a folder name like 'no_tumor' into a readable label like 'No Tumor'."""
    return _DISPLAY_NAMES.get(class_name.lower(), class_name.replace("_", " ").title())


def build_model(num_classes):
    """
    Build the CNN.

    Input  : 128 x 128 grayscale image (values 0-255)
    Output : probability for each class

    Three blocks of Conv -> BatchNorm -> MaxPool learn image features
    (edges, textures, shapes). A final convolution called "last_conv" holds the
    feature maps that Grad-CAM uses to draw the heatmap.
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="mri_input")
    x = layers.Rescaling(1.0 / 255)(inputs)                  # scale pixels to 0-1

    for filters in (32, 64, 128):                            # 128 -> 64 -> 32 -> 16 pixels
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name=LAST_CONV_LAYER)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    logits = layers.Dense(num_classes, name=LOGITS_LAYER)(x)  # raw scores
    probs = layers.Softmax(name="probs")(logits)              # scores -> probabilities
    return keras.Model(inputs, probs, name="mediscan_cnn")


def load_trained_model():
    """
    Load the trained model from the models/ folder.

    Returns (model, class_names, info, message)
      - model is None when no trained model is available; `message` explains why.
    """
    if not os.path.exists(MODEL_PATH):
        return None, DEFAULT_CLASS_NAMES, {}, (
            f"Trained model file not found at: {MODEL_PATH}"
        )

    info = {}
    if os.path.exists(INFO_PATH):
        try:
            with open(INFO_PATH) as f:
                info = json.load(f)
        except Exception:
            info = {}

    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        model.get_layer(LAST_CONV_LAYER)          # make sure Grad-CAM can work with it
        model.get_layer(LOGITS_LAYER)
    except Exception as error:
        return None, DEFAULT_CLASS_NAMES, {}, f"Model file could not be loaded: {error}"

    class_names = info.get("class_names", DEFAULT_CLASS_NAMES)
    if model.output_shape[-1] != len(class_names):
        return None, DEFAULT_CLASS_NAMES, {}, (
            "The model and model_info.json disagree about the number of classes."
        )
    return model, class_names, info, "Model loaded."


def build_demo_model():
    """
    DEMO MODE ONLY: the same network with UNTRAINED (random) weights.
    It lets the interface and the stability pipeline be shown when no trained
    model exists. Its outputs have NO medical meaning and the app labels them so.
    """
    keras.utils.set_random_seed(0)
    return build_model(len(DEFAULT_CLASS_NAMES))


def predict_probs(model, batch):
    """Run the model on a batch of preprocessed images. Returns an (N, classes) array."""
    return model.predict(batch, verbose=0)
