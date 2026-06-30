"""
hand_features.py
-----------------
Shared MediaPipe hand-landmark extraction used by collection, training,
webcam recognition, and static-image recognition.

Uses MediaPipe's current Tasks API (HandLandmarker), not the older
`mp.solutions.hands` API, which is being phased out and is unavailable in
some current MediaPipe installs.

Why landmarks instead of raw pixels?
  - No GPU needed, trains in seconds on a laptop CPU.
  - Works for both static images and live webcam with the exact same pipeline.
  - Needs far less training data than a CNN (a few dozen samples per sign is
    enough for a usable demo, since MediaPipe already did the hard visual work).

Before using this module, run download_model.py once to fetch the model file.
"""

import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "hand_landmarker.task")

# Standard 21-point hand connectivity (MediaPipe's HAND_CONNECTIONS), hardcoded
# since the legacy mp.solutions.drawing_utils helper may not be available.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]


def _check_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Hand landmark model not found at {MODEL_PATH}. Run `python download_model.py` first."
        )


def make_landmarker(mode="VIDEO", num_hands=1, min_detection_confidence=0.6):
    """
    mode: "IMAGE" for single static photos, "VIDEO" for webcam frame streams.
    """
    _check_model()
    running_mode = {
        "IMAGE": vision.RunningMode.IMAGE,
        "VIDEO": vision.RunningMode.VIDEO,
    }[mode]

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks(landmarker, frame_rgb, mode="VIDEO", timestamp_ms=0):
    """
    Returns (feature_vector, raw_landmark_list_or_None).
    feature_vector is a normalized 63-dim array (21 landmarks x,y,z) for the
    first detected hand, or None if no hand is found.

    Normalization: translate so the wrist (landmark 0) is the origin, then
    scale so the hand's extent has unit size. This makes features roughly
    invariant to hand position/distance from camera.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    if mode == "IMAGE":
        result = landmarker.detect(mp_image)
    else:
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks:
        return None, None

    hand = result.hand_landmarks[0]  # first detected hand
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand])

    wrist = pts[0]
    pts = pts - wrist

    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale

    return pts.flatten(), hand


def draw_landmarks(frame_bgr, hand_landmarks):
    """hand_landmarks: the raw (un-normalized) landmark list returned by extract_landmarks."""
    if hand_landmarks is None:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame_bgr, points[start], points[end], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame_bgr, (x, y), 4, (0, 0, 255), -1)

    return frame_bgr
