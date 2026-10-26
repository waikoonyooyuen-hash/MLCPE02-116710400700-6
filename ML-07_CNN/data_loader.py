import os
import cv2
import numpy as np

from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def read_image_unicode(path):
    """Read an image even when the Windows path contains Thai/Unicode characters."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_data(data_path, img_size=100, max_per_class=None):
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(data_path)}"
        )

    images = []
    labels = []

    classes = sorted(
        folder for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
        and not folder.startswith(".")
    )

    print("Detected classes:", classes)

    if len(classes) != 2:
        raise ValueError(
            f"This lab expects exactly 2 classes, but found {len(classes)}: {classes}"
        )

    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        loaded = 0
        skipped = 0

        for filename in filenames:
            if max_per_class is not None and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)
            image = read_image_unicode(image_path)

            if image is None:
                print(f"Skipped unreadable image: {filename}")
                skipped += 1
                continue

            try:
                image = preprocess_image(image, img_size)
            except Exception as exc:
                print(f"Skipped image {filename}: {exc}")
                skipped += 1
                continue

            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {class_name}: {loaded} images ({skipped} skipped)")

    if not images:
        raise ValueError("No valid images were loaded from the dataset.")

    # Make sure both classes contain data before stratified splitting.
    counts = np.bincount(np.asarray(labels), minlength=len(classes))
    if np.any(counts < 2):
        raise ValueError(f"Not enough images for both classes: {dict(zip(classes, counts))}")

    return np.stack(images), np.asarray(labels, dtype=np.int32), classes
