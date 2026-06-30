"""
recognize_webcam.py
--------------------
Real-time sign recognition from your webcam using the model trained by
train_model.py.

Usage:
    python recognize_webcam.py

Press 'q' to quit.
"""

import os
import time
import collections
import cv2
import joblib

from hand_features import make_landmarker, extract_landmarks, draw_landmarks

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "sign_model.joblib")


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run train_model.py first.")

    clf = joblib.load(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    landmarker = make_landmarker(mode="VIDEO")
    start_time = time.time()

    # Smooth predictions over a short window so the label doesn't flicker frame to frame.
    recent_preds = collections.deque(maxlen=8)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((time.time() - start_time) * 1000)
        features, hand = extract_landmarks(landmarker, rgb, mode="VIDEO", timestamp_ms=timestamp_ms)
        frame = draw_landmarks(frame, hand)

        if features is not None:
            pred = clf.predict([features])[0]
            proba = clf.predict_proba([features])[0].max()
            recent_preds.append(pred)
            stable_pred = max(set(recent_preds), key=recent_preds.count)

            cv2.putText(frame, f"{stable_pred} ({proba:.0%})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            recent_preds.clear()
            cv2.putText(frame, "No hand detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Sign Language Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
