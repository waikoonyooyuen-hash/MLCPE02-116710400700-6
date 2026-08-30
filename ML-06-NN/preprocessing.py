import numpy as np

def preprocess_images(images):
    # Convert pixel values from 0-16 to 0-1
    return np.asarray(images, dtype=np.float32) / 16.0
