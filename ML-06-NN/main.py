import os
from data_loader import load_data
from preprocessing import preprocess_images
from split_data import split_dataset
from nn_model import train_model, predict_model
from evaluate import evaluate_model, plot_training

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 50)
    print("Neural Network: Handwritten Digits Recognition")
    print("=" * 50)

    print("\n[Step 1] Loading dataset...")
    X, y, classes = load_data()

    print("\n[Step 2] Preprocessing...")
    X = preprocess_images(X)

    print("\n[Step 3] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print("Training samples  :", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples   :", len(X_test))

    print("\n[Step 4] Training Neural Network...")
    model = train_model(X_train, y_train, OUTPUT_DIR)

    print("\n[Step 5] Prediction...")
    predictions = predict_model(model, X_test)

    print("\n[Step 6] Evaluation...")
    accuracy = evaluate_model(
        y_test, predictions, classes,
        f"{OUTPUT_DIR}/confusion_matrix.png"
    )
    plot_training(model, f"{OUTPUT_DIR}/training_history.png")
    print(f"\nFinal Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
