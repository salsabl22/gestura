import os
import urllib.request

from src.config import MODEL_PATH, MEDIAPIPE_MODEL_URL


def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Downloading MediaPipe hand landmarker model...")
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
