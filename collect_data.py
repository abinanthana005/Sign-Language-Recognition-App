"""
collect_data.py
----------------
Webcam tool for building your own small training set.

Usage:
    python download_model.py   # one-time setup
    python collect_data.py

Controls:
    - Type a letter/word in the terminal prompt to start a "recording session" for that label.
    - Press SPACE in the video window to capture one sample of your current hand pose.
    - Press 'n' to finish this label and move to the next one.
    - Press 'q' to quit and save everything to data/landmarks.csv

Tips for a usable demo with minimal data:
    - Capture ~30-50 samples per sign, slightly varying hand angle/distance each time.
    - Keep lighting consistent and your hand fully in frame.
    - For a quick first demo, start with just 5-10 distinct signs rather than the full alphabet.
"""

import csv
import os
import time
import cv2

from hand_features import make_landmarker, extract_landmarks, draw_landmarks

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "landmarks.csv")


def main():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    file_exists = os.path.isfile(DATA_PATH)

    cap = cv2.VideoCapture(0)
    landmarker = make_landmarker(mode="VIDEO")
    start_time = time.time()

    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["label"] + [f"f{i}" for i in range(63)])

        while True:
            label = input("\nEnter label to record (or 'quit' to finish): ").strip()
            if label.lower() == "quit":
                break

            count = 0
            print(f"Recording '{label}'. SPACE = capture, n = next label, q = quit entirely.")

            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Could not read from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int((time.time() - start_time) * 1000)
                features, hand = extract_landmarks(landmarker, rgb, mode="VIDEO", timestamp_ms=timestamp_ms)
                frame = draw_landmarks(frame, hand)

                cv2.putText(frame, f"Label: {label}  Samples: {count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Collect Sign Data", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and features is not None:
                    writer.writerow([label] + features.tolist())
                    count += 1
                    print(f"  captured sample #{count}")
                elif key == ord("n"):
                    break
                elif key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    print(f"Saved data to {DATA_PATH}")
                    return

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved data to {DATA_PATH}")


if __name__ == "__main__":
    main()
