import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def evaluate_model(y_test, predictions, classes, save_path):
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=classes))
    print("Confusion Matrix:")
    matrix = confusion_matrix(y_test, predictions)
    print(matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix)
    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i,j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return accuracy

def plot_training(model, save_path):
    if hasattr(model, "loss_curve_"):
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(model.loss_curve_)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
