import numpy as np


def to_features(images):
    """Convert (n, 8, 8) images into feature vectors and normalize to 0-1."""
    features = images.reshape(len(images), -1).astype(np.float32)
    features /= 16.0
    return features
