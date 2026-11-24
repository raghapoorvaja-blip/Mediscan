"""
stability.py
------------
FEATURE 2: Evidence Stability Verification.

Idea
----
A trustworthy AI should not change its mind (or look at completely different
parts of the image) when the scan is changed in tiny, harmless ways, such as a
slightly brighter picture or a tiny bit of noise.

Steps
-----
1. make_variations()        : create 8 slightly modified copies of the MRI
2. run_stability_analysis() : for each copy
       - run the SAME model                       -> prediction + confidence
       - make a Grad-CAM heatmap (same class)     -> explanation
       - compare with the original prediction / heatmap
3. Combine everything into ONE Evidence Stability Score (0-100)

The score measures how CONSISTENT the model is.
It does NOT measure whether the model is medically CORRECT.
"""

import cv2
import numpy as np

from gradcam import compute_gradcam, overlay_heatmap
from model import predict_probs
from preprocessing import preprocess_batch

# ----------------------------------------------------------------------------
# Score settings (easy to change and easy to explain)
# ----------------------------------------------------------------------------
WEIGHT_PREDICTION = 0.40     # do the predicted classes stay the same?
WEIGHT_CONFIDENCE = 0.20     # does the confidence stay about the same?
WEIGHT_HEATMAP = 0.40        # does the heatmap keep highlighting the same areas?

HIGH_THRESHOLD = 80          # score >= 80  -> High stability
MODERATE_THRESHOLD = 60      # score 60-79  -> Moderate stability, below 60 -> Low


# ----------------------------------------------------------------------------
# Step 1: controlled image variations
# ----------------------------------------------------------------------------
def _rotate(image, angle):
    """Rotate around the image centre. Empty corners are filled with black."""
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def _brightness(image, amount):
    return np.clip(image.astype(np.float32) + amount, 0, 255).astype(np.uint8)


def _contrast(image, factor):
    mean = image.mean()
    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def _noise(image, sigma, seed=42):
    rng = np.random.default_rng(seed)                  # fixed seed -> same result every run
    noisy = image.astype(np.float32) + rng.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _resize_blur(image, scale):
    """Shrink slightly then enlarge again (a small loss of detail, no movement)."""
    h, w = image.shape
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def make_variations(gray):
    """
    Create the controlled variations of the MRI.

    Each variation is a dict with:
      name        : short label
      description : what was changed
      image       : the modified grayscale image
      undo        : function that moves a heatmap back to the original orientation
                    (only needed for rotations) or None
    """
    return [
        {"name": "Brighter", "description": "Brightness +20", "image": _brightness(gray, 20), "undo": None},
        {"name": "Darker", "description": "Brightness -20", "image": _brightness(gray, -20), "undo": None},
        {"name": "More contrast", "description": "Contrast x1.15", "image": _contrast(gray, 1.15), "undo": None},
        {"name": "Less contrast", "description": "Contrast x0.85", "image": _contrast(gray, 0.85), "undo": None},
        {"name": "Noise", "description": "Gaussian noise (sigma 8)", "image": _noise(gray, 8), "undo": None},
        {"name": "Rotate +5", "description": "Rotated 5 degrees clockwise",
         "image": _rotate(gray, -5), "undo": lambda hm: _rotate(hm, 5)},
        {"name": "Rotate -5", "description": "Rotated 5 degrees anticlockwise",
         "image": _rotate(gray, 5), "undo": lambda hm: _rotate(hm, -5)},
        {"name": "Resize 85%", "description": "Shrunk to 85% and enlarged again",
         "image": _resize_blur(gray, 0.85), "undo": None},
    ]


# ----------------------------------------------------------------------------
# Step 2: comparing heatmaps
# ----------------------------------------------------------------------------
def heatmap_similarity(map_a, map_b):
    """
    Similarity of two heatmaps as a percentage (0-100).
    Uses Pearson correlation: 100 = identical pattern, 0 = unrelated or opposite.
    """
    a, b = map_a.flatten(), map_b.flatten()
    a_flat, b_flat = a.std() < 1e-6, b.std() < 1e-6
    if a_flat and b_flat:
        return 100.0            # both maps are empty -> they agree
    if a_flat or b_flat:
        return 0.0              # one is empty, the other is not -> they disagree
    correlation = float(np.corrcoef(a, b)[0, 1])
    return max(0.0, correlation) * 100.0


# ----------------------------------------------------------------------------
# Step 3: classify the final score
# ----------------------------------------------------------------------------
def classify_score(score):
    """Turn the 0-100 score into a status label, colour key and plain-English message."""
    if score >= HIGH_THRESHOLD:
        return "High Stability", "high", "Reliable-looking AI evidence."
    if score >= MODERATE_THRESHOLD:
        return "Moderate Stability", "moderate", "Requires additional review."
    return "Low Stability", "low", "Unstable AI evidence - doctor review recommended."


# ----------------------------------------------------------------------------
# The main function
# ----------------------------------------------------------------------------
def run_stability_analysis(model, class_names, gray, original_class, original_probs, original_heatmap):
    """
    Run the full Evidence Stability Verification.

    gray            : original grayscale MRI (2-D uint8)
    original_class  : class index predicted for the original image
    original_probs  : probabilities for the original image
    original_heatmap: Grad-CAM heatmap of the original image (128 x 128)

    Returns a dictionary with per-variation results and the overall scores.
    """
    variations = make_variations(gray)

    # Run all variations through the SAME model in one go
    batch = preprocess_batch([v["image"] for v in variations])
    all_probs = predict_probs(model, batch)

    results = []
    for index, (variation, probs) in enumerate(zip(variations, all_probs)):
        predicted = int(np.argmax(probs))

        # Heatmap for the ORIGINAL predicted class, so we ask the same question each time:
        # "where does the evidence for the original decision come from?"
        heatmap, _ = compute_gradcam(model, batch[index][np.newaxis], class_index=original_class)

        # Rotated images have rotated heatmaps: rotate them back before comparing
        aligned = variation["undo"](heatmap) if variation["undo"] is not None else heatmap

        similarity = heatmap_similarity(original_heatmap, aligned)
        confidence_change = abs(float(probs[original_class]) - float(original_probs[original_class])) * 100

        results.append({
            "name": variation["name"],
            "description": variation["description"],
            "image": variation["image"],
            "predicted_class": predicted,
            "predicted_name": class_names[predicted],
            "confidence": float(probs[predicted]) * 100,
            "probs": probs,
            "same_prediction": predicted == original_class,
            "confidence_change": confidence_change,        # change in % points for the original class
            "heatmap": aligned,
            "overlay": overlay_heatmap(variation["image"], heatmap),
            "heatmap_similarity": similarity,
        })

    # ---- Combine into the three summary numbers -----------------------------
    prediction_consistency = 100.0 * np.mean([r["same_prediction"] for r in results])
    confidence_stability = max(0.0, 100.0 - float(np.mean([r["confidence_change"] for r in results])))
    heatmap_consistency = float(np.mean([r["heatmap_similarity"] for r in results]))

    score = (WEIGHT_PREDICTION * prediction_consistency
             + WEIGHT_CONFIDENCE * confidence_stability
             + WEIGHT_HEATMAP * heatmap_consistency)
    score = float(np.clip(score, 0, 100))

    status, status_key, message = classify_score(score)
    return {
        "variations": results,
        "prediction_consistency": float(prediction_consistency),
        "confidence_stability": confidence_stability,
        "heatmap_consistency": heatmap_consistency,
        "score": score,
        "status": status,
        "status_key": status_key,
        "message": message,
    }
