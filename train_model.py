"""
train_model.py
---------------
Trains a RandomForest classifier on the landmark data collected by
collect_data.py. Trains in seconds on CPU; no GPU needed.

Usage:
    python train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "landmarks.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "sign_model.joblib")


def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"No data found at {DATA_PATH}. Run collect_data.py first to record some signs."
        )

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    if df["label"].nunique() < 2:
        raise ValueError("Need at least 2 distinct labels to train a classifier.")

    # Stratify only if every class has enough samples for a test split
    min_class_count = df["label"].value_counts().min()
    stratify = y if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    if len(X_test) > 0:
        preds = clf.predict(X_test)
        print(classification_report(y_test, preds))

    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Classes: {sorted(set(y))}")


if __name__ == "__main__":
    main()
