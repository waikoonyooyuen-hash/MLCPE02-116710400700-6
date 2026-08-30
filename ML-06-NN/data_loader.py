from sklearn.datasets import load_digits
import numpy as np

def load_data():
    digits = load_digits()
    X = digits.images
    y = digits.target
    classes = [str(i) for i in sorted(np.unique(y))]
    print("Detected classes:", classes)
    print("Total images:", len(X))
    return X, y, classes
