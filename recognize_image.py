"""
recognize_image.py
-------------------
Classify a sign from a single static image file using the model trained
by train_model.py.

Usage:
    python recognize_image.py path/to/photo.jpg
"""

import os
import sys
import cv2
import joblib

from hand_features import make_landmarker, extract_landmarks

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "sign_model.joblib")


def predict_image(image_path: str):
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run train_model.py first.")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No image found at {image_path}")

    clf = joblib.load(MODEL_PATH)
    landmarker = make_landmarker(mode="IMAGE")

    frame = cv2.imread(image_path)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    features, _ = extract_landmarks(landmarker, rgb, mode="IMAGE")

    if features is None:
        print("No hand detected in this image.")
        return None

    pred = clf.predict([features])[0]
    proba = clf.predict_proba([features])[0].max()
    print(f"Predicted sign: {pred}  (confidence: {proba:.0%})")
    return pred


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python recognize_image.py path/to/photo.jpg")
        sys.exit(1)
    predict_image(sys.argv[1])
