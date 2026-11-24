"""
gradcam.py
----------
Grad-CAM = "Gradient-weighted Class Activation Mapping".

Plain-English idea
------------------
1. Run the image through the CNN and look at the feature maps of the last
   convolution layer ("last_conv").
2. Ask: "If this feature map became a bit stronger, would the score of the
   predicted class go up?"  (that is the gradient)
3. Feature maps that push the score up get a large weight, the others a small one.
4. Add the feature maps together using those weights -> a coarse picture showing
   WHERE in the image the evidence for the prediction came from.

The heatmap is computed from the real model for every image. Nothing is faked.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import LAST_CONV_LAYER, LOGITS_LAYER
from preprocessing import IMG_SIZE

_grad_model_cache = {}   # building the helper model once per model saves time


def _get_grad_model(model):
    """A helper model that returns BOTH the last conv feature maps and the raw class scores."""
    key = id(model)
    if key not in _grad_model_cache:
        _grad_model_cache[key] = keras.Model(
            model.inputs,
            [model.get_layer(LAST_CONV_LAYER).output, model.get_layer(LOGITS_LAYER).output],
        )
    return _grad_model_cache[key]


def compute_gradcam(model, image_batch, class_index=None):
    """
    Compute a Grad-CAM heatmap for ONE image.

    image_batch : array of shape (1, 128, 128, 1)
    class_index : which class to explain. None = the model's own predicted class.

    Returns (heatmap, class_index)
      heatmap : 128 x 128 float array between 0 (no influence) and 1 (strongest influence)
    """
    grad_model = _get_grad_model(model)
    x = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        feature_maps, logits = grad_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(logits[0]))
        class_score = logits[:, class_index]

    # How strongly does each feature map influence the class score?
    gradients = tape.gradient(class_score, feature_maps)
    weights = tf.reduce_mean(gradients, axis=(1, 2))                       # one weight per feature map

    # Weighted sum of feature maps; keep only positive influence (ReLU)
    cam = tf.reduce_sum(feature_maps * weights[:, tf.newaxis, tf.newaxis, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()

    # Enlarge to image size and scale to the 0-1 range
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0, None)
    if cam.max() > 1e-8:
        cam = cam / cam.max()
    else:
        cam = np.zeros_like(cam)       # model found no positive evidence: honest blank map
    return cam.astype(np.float32), class_index


def heatmap_to_rgb(heatmap, size=None):
    """Colour a 0-1 heatmap (blue = low influence, red = high influence)."""
    if size is not None:
        heatmap = cv2.resize(heatmap, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    heat_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    coloured = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


def overlay_heatmap(gray_image, heatmap, max_strength=0.6):
    """
    Lay the coloured heatmap over the grayscale MRI.
    Areas with little influence stay clear so the anatomy remains visible.
    """
    height, width = gray_image.shape
    heat_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    heat_resized = np.clip(heat_resized, 0, 1)

    base = np.stack([gray_image] * 3, axis=-1).astype(np.float32)
    colour = heatmap_to_rgb(heat_resized).astype(np.float32)
    strength = (heat_resized * max_strength)[..., np.newaxis]     # stronger heat -> more colour
    blended = base * (1 - strength) + colour * strength
    return blended.astype(np.uint8)
