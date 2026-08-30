from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm(X_train, y_train, pca_components=40):
    """Train an RBF SVM after scaling and reducing the feature size."""
    n_components = min(pca_components, X_train.shape[0], X_train.shape[1])

    scaler = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, whiten=True, random_state=42)),
    ])

    X_train_scaled = scaler.fit_transform(X_train)

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_svm(model, scaler, X_test):
    X_test_scaled = scaler.transform(X_test)
    return model.predict(X_test_scaled)
