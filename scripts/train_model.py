import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DATASET_FILE,
    ML_MODEL_PATH,
    RANDOM_FOREST_ESTIMATORS,
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_RANDOM_STATE,
)


def load_dataset() -> tuple:
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"Dataset not found at '{DATASET_FILE}'. "
            "Run scripts/collect_dataset.py first."
        )
    df = pd.read_csv(DATASET_FILE)
    print(f"Total samples: {len(df)}")
    print(f"\nLabel distribution:\n{df['label'].value_counts().sort_index()}\n")
    X = df.drop("label", axis=1).values
    y = df["label"].values
    return X, y


def train(X, y) -> tuple:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=RANDOM_FOREST_RANDOM_STATE,
        stratify=y_encoded,
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")
    print("Training model...")

    model = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_ESTIMATORS,
        max_depth=RANDOM_FOREST_MAX_DEPTH,
        random_state=RANDOM_FOREST_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean() * 100
    print(f"Accuracy: {accuracy:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    return model, label_encoder


def save_model(model, label_encoder):
    os.makedirs(os.path.dirname(ML_MODEL_PATH), exist_ok=True)
    with open(ML_MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "label_encoder": label_encoder}, f)
    print(f"Model saved: {ML_MODEL_PATH}")


def main():
    print("=== BISINDO MODEL TRAINER ===\n")
    X, y = load_dataset()
    model, label_encoder = train(X, y)
    save_model(model, label_encoder)
    print("\nDone. Run main.py to use the trained model.")


if __name__ == "__main__":
    main()
