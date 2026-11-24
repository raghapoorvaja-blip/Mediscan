"""
train.py
--------
Trains the brain-MRI classifier and saves it to models/.

Usage:
    python train.py --data_dir dataset --epochs 25

Expected dataset layout (one sub-folder per class):
    dataset/
        Training/
            glioma_tumor/      *.jpg
            meningioma_tumor/  *.jpg
            no_tumor/          *.jpg
            pituitary_tumor/   *.jpg
        Testing/
            (same four folders)

See README.md for where to download the dataset.
"""

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from model import INFO_PATH, MODEL_DIR, MODEL_PATH, build_model
from preprocessing import IMG_SIZE, preprocess_gray

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_folder(folder, class_names):
    """Read every image in folder/<class>/ and return (images, labels)."""
    images, labels = [], []
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder, class_name)
        if not os.path.isdir(class_folder):
            raise SystemExit(f"Missing folder: {class_folder}")
        for file_name in sorted(os.listdir(class_folder)):
            if not file_name.lower().endswith(IMAGE_TYPES):
                continue
            gray = cv2.imread(os.path.join(class_folder, file_name), cv2.IMREAD_GRAYSCALE)
            if gray is None:                       # skip unreadable files
                continue
            images.append(preprocess_gray(gray))   # SAME preprocessing as the app uses
            labels.append(label)
    return np.stack(images), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    keras.utils.set_random_seed(42)
    train_dir = os.path.join(args.data_dir, "Training")
    test_dir = os.path.join(args.data_dir, "Testing")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise SystemExit(f"Could not find {train_dir} and {test_dir}. See README.md for dataset setup.")

    class_names = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    print("Classes:", class_names)

    print("Loading images ...")
    x_all, y_all = load_folder(train_dir, class_names)
    x_test, y_test = load_folder(test_dir, class_names)
    print(f"Training images: {len(x_all)}   Test images: {len(x_test)}")

    # Hold back 15% of the training images (same share from every class) to watch for over-fitting
    rng = np.random.default_rng(42)
    val_idx = []
    for label in range(len(class_names)):
        idx = np.where(y_all == label)[0]
        rng.shuffle(idx)
        val_idx.extend(idx[: int(0.15 * len(idx))])
    val_mask = np.zeros(len(y_all), dtype=bool)
    val_mask[val_idx] = True
    x_train, y_train = x_all[~val_mask], y_all[~val_mask]
    x_val, y_val = x_all[val_mask], y_all[val_mask]

    # Data augmentation: small random changes during training only.
    # This teaches the model to ignore tiny differences in brightness, contrast,
    # rotation and noise, which is exactly what the stability check tests later.
    augment = keras.Sequential([
        layers.RandomRotation(0.04),                       # about +/- 14 degrees at most
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.1, value_range=(0, 255)),
    ])

    def augment_batch(x, y):
        x = augment(x, training=True)
        x = x + tf.random.normal(tf.shape(x), stddev=4.0)      # small random noise (0-255 scale)
        return tf.clip_by_value(x, 0.0, 255.0), y

    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(2000, seed=42).batch(args.batch_size)
                .map(augment_batch)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size)

    # Give rarer classes a larger weight so the model does not ignore them
    counts = np.bincount(y_train)
    class_weight = {i: float(len(y_train) / (len(counts) * c)) for i, c in enumerate(counts)}

    model = build_model(len(class_names))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = [
        # save the best model after every epoch so progress is never lost
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        class_weight=class_weight, callbacks=callbacks, verbose=2)
    print(f"Training took {(time.time() - start) / 60:.1f} minutes")

    # ---- Final honest check on images the model has never seen --------------
    test_probs = model.predict(x_test, verbose=0)
    test_pred = test_probs.argmax(axis=1)
    test_acc = float((test_pred == y_test).mean())
    val_acc = float((model.predict(x_val, verbose=0).argmax(axis=1) == y_val).mean())
    matrix = tf.math.confusion_matrix(y_test, test_pred, num_classes=len(class_names)).numpy()
    print(f"\nTEST accuracy: {test_acc * 100:.1f}%   (validation accuracy {val_acc * 100:.1f}%)")
    print("Confusion matrix (rows = true class, columns = predicted):")
    print(class_names)
    print(matrix)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    info = {
        "class_names": class_names,
        "image_size": IMG_SIZE,
        "test_accuracy": round(test_acc, 4),
        "validation_accuracy": round(val_acc, 4),
        "epochs_run": len(history.history["loss"]),
        "training_images": int(len(x_train)),
        "test_images": int(len(x_test)),
        "confusion_matrix": matrix.tolist(),
        "trained_on": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
