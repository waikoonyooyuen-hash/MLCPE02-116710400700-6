from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CSV_PATH = Path(__file__).resolve().parents[1] / "data-cars" / "car data.csv"

TARGET = "Transmission"

numeric_features = [
    "Year",
    "Selling_Price",
    "Present_Price",
    "Kms_Driven",
    "Owner",
]


def load_data():

    # -----------------------------
    # 1. Load CSV
    # -----------------------------
    df = pd.read_csv(CSV_PATH)

    df = df.dropna(subset=[TARGET]).copy()

    # -----------------------------
    # 2. X = features
    #    y = target
    # -----------------------------
    X = df[numeric_features].astype(float).to_numpy()

    labels = df[TARGET].astype(str).to_numpy()

    # Manual / Automatic -> 0 / 1
    class_names = sorted(np.unique(labels).tolist())

    class_to_id = {
        name: i
        for i, name in enumerate(class_names)
    }

    y = np.array(
        [class_to_id[label] for label in labels],
        dtype=np.int32
    )

    # -----------------------------
    # 3. Split Train / Validation / Test
    # -----------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # -----------------------------
    # 4. Scale features
    # -----------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # -----------------------------
    # 5. Return dictionary
    # -----------------------------
    return {
        "X_train": X_train,
        "y_train": y_train,

        "X_val": X_val,
        "y_val": y_val,

        "X_test": X_test,
        "y_test": y_test,

        "class_names": class_names,

        "n_rows": len(df),

        "feature_names": numeric_features,
    }