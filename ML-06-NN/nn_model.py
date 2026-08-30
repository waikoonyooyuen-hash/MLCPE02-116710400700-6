from sklearn.neural_network import MLPClassifier
import joblib

def build_model():
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=30,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5
    )

def train_model(X_train, y_train, output_dir):
    model = build_model()
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    joblib.dump(model, f"{output_dir}/nn_model.pkl")
    return model

def predict_model(model, X_test):
    return model.predict(X_test.reshape(len(X_test), -1))
