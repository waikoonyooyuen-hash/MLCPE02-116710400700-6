from sklearn.datasets import load_digits
import numpy as np


def load_data():
    """Load the built-in handwritten digits dataset from scikit-learn."""
    digits = load_digits()

    # Images are 8x8 grayscale images. Pixel values are 0-16.
    images = digits.images.astype(np.float32)
    labels = digits.target.astype(np.int64)

    classes = [str(i) for i in range(10)]

    return images, labels, classes
