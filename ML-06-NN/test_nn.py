import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from joblib import load

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def test_nn():
    digits = load_digits()
    model = load(f"{OUTPUT_DIR}/nn_model.pkl")
    indices = np.array([0, 100, 500, 1000])
    X = digits.images[indices] / 16.0
    y = digits.target[indices]
    pred = model.predict(X.reshape(len(X), -1))

    for i in range(len(X)):
        print(f"[{i+1}] Pred: {pred[i]}  True: {y[i]}  {'OK' if pred[i] == y[i] else 'WRONG'}")

    fig, axes = plt.subplots(2,2, figsize=(6,6))
    for ax, image, p, t in zip(axes.ravel(), X, pred, y):
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Pred: {p} | True: {t}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/prediction_sample.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    test_nn()
