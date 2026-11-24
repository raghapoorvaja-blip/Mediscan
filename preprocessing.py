"""
preprocessing.py
----------------
Everything that happens to an image BEFORE it reaches the AI model:

1. validate_and_load()  -> checks the uploaded file is a usable image
2. preprocess_gray()    -> resizes a grayscale image to the size the model expects

IMPORTANT: train.py uses the same preprocess_gray() function, so the model sees
exactly the same kind of input during training and during real use.
"""

import io

import cv2
import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 128                      # the model works on 128 x 128 images
MAX_FILE_MB = 10                    # reject files larger than this
MAX_DISPLAY_SIDE = 512              # big images are shrunk to keep the app fast
MIN_SIDE = 64                       # images smaller than this are rejected
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def validate_and_load(file_bytes, filename):
    """
    Check an uploaded file and convert it to a grayscale NumPy array.

    Returns (gray_image, warnings, error)
      - gray_image : 2-D uint8 array (0-255), or None if the file is invalid
      - warnings   : list of friendly warning strings (image is usable but unusual)
      - error      : error message string, or None if everything is fine
    """
    warnings = []

    # --- Check 1: file extension -------------------------------------------
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        return None, warnings, (
            "Unsupported file type. Please upload a PNG, JPG, BMP, TIFF or WEBP image."
        )

    # --- Check 2: file size -------------------------------------------------
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return None, warnings, f"File is too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_MB} MB."
    if len(file_bytes) == 0:
        return None, warnings, "The uploaded file is empty."

    # --- Check 3: is it a real, readable image? -----------------------------
    try:
        probe = Image.open(io.BytesIO(file_bytes))
        probe.verify()                              # raises an error if the file is corrupt
        image = Image.open(io.BytesIO(file_bytes))  # verify() closes the file, so re-open
        image = ImageOps.exif_transpose(image)      # respect camera/scanner rotation flags
    except Exception:
        return None, warnings, "This file could not be read as an image. It may be corrupt."

    # --- Check 4: image dimensions -----------------------------------------
    width, height = image.size
    if min(width, height) < MIN_SIDE:
        return None, warnings, (
            f"Image is too small ({width}x{height}). Minimum side length is {MIN_SIDE} pixels."
        )

    # --- Check 5: colour image? MRI scans are normally grayscale ------------
    rgb = np.array(image.convert("RGB")).astype(np.int16)
    colour_difference = np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()
    if colour_difference > 12:
        warnings.append(
            "This looks like a colour image. Brain MRI scans are normally grayscale, "
            "so the result may not be meaningful. It was converted to grayscale."
        )

    # Convert to grayscale NumPy array
    gray = np.array(image.convert("L"), dtype=np.uint8)

    # --- Check 6: blank image? ----------------------------------------------
    if gray.std() < 5:
        return None, warnings, "The image looks blank (almost no variation in brightness)."

    # --- Shrink very large images so everything stays fast ------------------
    longest = max(gray.shape)
    if longest > MAX_DISPLAY_SIDE:
        scale = MAX_DISPLAY_SIDE / longest
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return gray, warnings, None


def preprocess_gray(gray):
    """
    Turn a grayscale image (2-D uint8) into model input of shape (128, 128, 1).
    Pixel values stay in the 0-255 range because the model rescales them itself.
    """
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32).reshape(IMG_SIZE, IMG_SIZE, 1)


def preprocess_batch(gray_images):
    """Preprocess a list of grayscale images into one batch of shape (N, 128, 128, 1)."""
    return np.stack([preprocess_gray(g) for g in gray_images], axis=0)
