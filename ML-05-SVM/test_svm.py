import json
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "output"
N_SAMPLES = 4


def test_svm(n_samples=N_SAMPLES):
    model = joblib.load(f"{OUTPUT_DIR}/svm_model.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")

    with open(f"{OUTPUT_DIR}/classes.json", encoding="utf-8") as f:
        classes = json.load(f)

    n_samples = min(n_samples, len(X_test))
    rng = np.random.default_rng(42)
    index = rng.choice(len(X_test), n_samples, replace=False)

    X_sample = X_test[index]
    y_sample = y_test[index]
    predictions = model.predict(scaler.transform(X_sample))

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]

        ax.imshow(X_sample[i].reshape(8, 8), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred: {pred} | True: {true}\n"
                     f"{'OK' if correct else 'WRONG'}")

        print(
            f"[{i + 1}] Pred: {pred:<2} True: {true:<2} "
            f"{'OK' if correct else 'WRONG'}"
        )

    correct_total = int((predictions == y_sample).sum())
    print(f"\nCorrect: {correct_total}/{n_samples}")

    fig.suptitle(f"SVM Prediction: {correct_total}/{n_samples} correct")
    fig.tight_layout()
    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_svm()
