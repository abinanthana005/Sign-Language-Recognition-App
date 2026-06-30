# Sign Language Recognition App (Python + MediaPipe)

A lightweight, beginner-friendly sign recognition app that works for both
**live webcam video** and **static images**, using the same pipeline for both.

## Why this approach (no big pretrained CNN needed)

Instead of classifying raw pixels (which needs a large labeled image dataset
and a GPU), this app uses **MediaPipe Hands** to extract 21 hand-joint
landmarks per frame, then trains a small **RandomForest classifier** on those
landmark positions. Benefits:

- Trains in seconds on a normal laptop CPU.
- Needs only ~30-50 example captures per sign to get a usable demo.
- The exact same feature extraction works for webcam frames and static photos.

The tradeoff: this approach handles **static hand poses** well (most of the
ASL alphabet, single-word signs). It does **not** capture motion-based signs
(like the letters J and Z, which involve movement) — that would need a
sequence model (e.g. an LSTM over landmark sequences), which is a reasonable
v2 upgrade once the basic pipeline is working.

## Setup

```bash
pip install mediapipe opencv-python scikit-learn pandas joblib
python download_model.py   # one-time: downloads the hand landmark model (~10MB)
```

> Note: this uses MediaPipe's current **Tasks API** (`HandLandmarker`), not the
> older `mp.solutions.hands` API you'll see in a lot of older tutorials —
> that legacy API isn't available in current MediaPipe installs on some platforms.

## Workflow

### 1. Collect training data (your own webcam)
```bash
python collect_data.py
```
Type a label (e.g. `A`, `hello`, `thanks`), press SPACE repeatedly to capture
samples of that sign from slightly different angles, press `n` for the next
label, `q` when done. Aim for 5-10 signs and ~30+ samples each for a first demo.

### 2. Train the model
```bash
python train_model.py
```
Prints accuracy on a held-out test split and saves `data/sign_model.joblib`.

### 3a. Recognize live via webcam
```bash
python recognize_webcam.py
```

### 3b. Recognize a static photo
```bash
python recognize_image.py path/to/photo.jpg
```

## Files
- `download_model.py` — one-time download of the MediaPipe hand landmark model
- `hand_features.py` — shared MediaPipe landmark extraction + normalization
- `collect_data.py` — webcam data collection tool
- `train_model.py` — trains and saves the classifier
- `recognize_webcam.py` — real-time recognition with prediction smoothing
- `recognize_image.py` — single-image recognition
- `data/landmarks.csv` — your collected training data (created on first run)
- `data/sign_model.joblib` — trained model (created by train_model.py)

## Scaling up later
- **Full ASL alphabet with public data**: download a labeled ASL alphabet
  image dataset (e.g. from Kaggle) and adapt `train_model.py` to read images,
  run them through `extract_landmarks`, and build the CSV that way instead of
  live webcam capture — same downstream training code works unchanged.
- **Motion-based signs (J, Z, full words/phrases)**: capture a short sequence
  of landmark frames per sign and train a small LSTM/GRU on the sequence
  instead of a single-frame RandomForest.
- **Two-hand signs**: set `max_num_hands=2` in `hand_features.py` and
  concatenate both hands' landmark vectors.
