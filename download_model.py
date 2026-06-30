"""
download_model.py
------------------
One-time setup: downloads the MediaPipe hand-landmark model file used by
hand_features.py. Run this once before anything else.

Usage:
    python download_model.py
"""

import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "hand_landmarker.task")


def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.isfile(MODEL_PATH):
        print(f"Model already present at {MODEL_PATH}")
        return
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
