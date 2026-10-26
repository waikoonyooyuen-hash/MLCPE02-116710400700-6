import json
import os

import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from cnn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
TEST_SIZE = 0.20
VAL_SIZE = 0.10
MAX_PER_CLASS = None
EPOCHS = 20
BATCH_SIZE = 16


def main():
    print("=" * 60)
    print("CNN Image Recognition: Rice vs Corn")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(
        DATA_PATH, IMG_SIZE, MAX_PER_CLASS
    )

    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
    with open(os.path.join(OUTPUT_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels
    np.save(os.path.join(OUTPUT_DIR, "features.npy"), X)
    print(f"Feature shape: {X.shape}")

    print("\n[Step 3] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    for name, value in [
        ("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
        ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)
    ]:
        np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), value)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    print("\n[Step 4] Training model...")
    model, history = train_model(
        X_train, y_train,
        X_val, y_val,
        len(classes),
        OUTPUT_DIR,
        EPOCHS,
        BATCH_SIZE
    )
    print("Training completed.")

    print("\n[Step 5] Testing model...")
    predictions = predict_model(model, X_test)

    print("\n[Step 6] Evaluating model...")
    evaluate_model(
        y_test,
        predictions,
        classes,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )
    plot_history(
        history,
        os.path.join(OUTPUT_DIR, "training_history.png")
    )

    print("\nDone. Check the outputs folder for the results.")


if __name__ == "__main__":
    main()
