import pickle
from typing import Optional

from src.config import ML_MODEL_PATH, ML_CONFIDENCE_THRESHOLD
from src.landmarks import (
    WRIST, THUMB_TIP, THUMB_MCP, THUMB_IP,
    INDEX_TIP, INDEX_PIP, INDEX_MCP, INDEX_DIP,
    MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP,
    RING_TIP, RING_PIP, RING_MCP,
    PINKY_TIP, PINKY_PIP, PINKY_MCP,
)
from src.gesture_utils import (
    landmark_distance,
    is_finger_up,
    is_finger_curled,
    is_thumb_up,
    are_fingers_touching,
    is_fist,
)

RecognitionResult = tuple[str, Optional[str], tuple[int, int, int], str]


def load_ml_model() -> tuple:
    try:
        with open(ML_MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        return data["model"], data["label_encoder"]
    except FileNotFoundError:
        return None, None


def _recognize_ml(landmarks, model, label_encoder) -> tuple[Optional[str], float]:
    if model is None:
        return None, 0.0
    row = []
    for lm in landmarks:
        row += [lm.x, lm.y, lm.z]
    prediction = model.predict([row])[0]
    confidence = model.predict_proba([row])[0].max()
    label = label_encoder.inverse_transform([prediction])[0]
    return label, confidence


def _recognize_bisindo_rules(lm) -> Optional[str]:
    idx_up = is_finger_up(lm, INDEX_TIP, INDEX_PIP)
    mid_up = is_finger_up(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring_up = is_finger_up(lm, RING_TIP, RING_PIP)
    pinky_up = is_finger_up(lm, PINKY_TIP, PINKY_PIP)
    thumb = is_thumb_up(lm)
    idx_curl = is_finger_curled(lm, INDEX_TIP, INDEX_MCP)
    mid_curl = is_finger_curled(lm, MIDDLE_TIP, MIDDLE_MCP)
    ring_curl = is_finger_curled(lm, RING_TIP, RING_MCP)
    pinky_curl = is_finger_curled(lm, PINKY_TIP, PINKY_MCP)

    if idx_curl and mid_curl and ring_curl and pinky_curl and not thumb:
        return "A"
    if idx_up and mid_up and ring_up and pinky_up and not thumb:
        return "B"
    if (
        not idx_curl and not mid_curl and not ring_curl and not pinky_curl
        and landmark_distance(lm[THUMB_TIP], lm[INDEX_TIP]) > 0.1
        and lm[INDEX_TIP].x < lm[THUMB_TIP].x
    ):
        return "C"
    if idx_up and mid_curl and ring_curl and pinky_curl and are_fingers_touching(lm, THUMB_TIP, MIDDLE_TIP):
        return "D"
    if (
        not idx_up and not mid_up and not ring_up and not pinky_up and not thumb
        and lm[INDEX_TIP].y > lm[INDEX_PIP].y
        and lm[MIDDLE_TIP].y > lm[MIDDLE_PIP].y
    ):
        return "E"
    if are_fingers_touching(lm, THUMB_TIP, INDEX_TIP) and mid_up and ring_up and pinky_up:
        return "F"
    if (
        idx_up and not mid_up and not ring_up and not pinky_up
        and abs(lm[INDEX_TIP].y - lm[INDEX_MCP].y) < 0.08
        and lm[INDEX_TIP].x > lm[INDEX_MCP].x
    ):
        return "G"
    if (
        idx_up and mid_up and not ring_up and not pinky_up
        and abs(lm[INDEX_TIP].y - lm[MIDDLE_TIP].y) < 0.05
    ):
        return "H"
    if not idx_up and not mid_up and not ring_up and pinky_up and not thumb:
        return "I"
    if not idx_up and not mid_up and not ring_up and pinky_up and thumb:
        return "J"
    if (
        idx_up and mid_up and not ring_up and not pinky_up
        and lm[THUMB_TIP].x > lm[INDEX_MCP].x
        and landmark_distance(lm[THUMB_TIP], lm[MIDDLE_PIP]) < 0.07
    ):
        return "K"
    if thumb and idx_up and not mid_up and not ring_up and not pinky_up and landmark_distance(lm[THUMB_TIP], lm[INDEX_TIP]) > 0.1:
        return "L"
    if (
        idx_curl and mid_curl and ring_curl and not pinky_up
        and lm[INDEX_TIP].y > lm[THUMB_TIP].y
        and lm[MIDDLE_TIP].y > lm[THUMB_TIP].y
    ):
        return "M"
    if (
        idx_curl and mid_curl and not ring_curl and not pinky_up
        and lm[INDEX_TIP].y > lm[THUMB_TIP].y
    ):
        return "N"
    if (
        are_fingers_touching(lm, THUMB_TIP, INDEX_TIP)
        and not mid_up and not ring_up and not pinky_up
        and landmark_distance(lm[THUMB_TIP], lm[INDEX_TIP]) < 0.04
    ):
        return "O"
    if idx_up and mid_up and not ring_up and not pinky_up and lm[INDEX_TIP].y > lm[WRIST].y and thumb:
        return "P"
    if (
        not mid_up and not ring_up and not pinky_up
        and lm[INDEX_TIP].y > lm[INDEX_MCP].y
        and lm[THUMB_TIP].y > lm[THUMB_MCP].y
    ):
        return "Q"
    if (
        idx_up and mid_up and not ring_up and not pinky_up
        and lm[INDEX_TIP].x > lm[MIDDLE_TIP].x
        and landmark_distance(lm[INDEX_TIP], lm[MIDDLE_TIP]) < 0.04
    ):
        return "R"
    if (
        idx_curl and mid_curl and ring_curl and pinky_curl and thumb
        and lm[THUMB_TIP].x < lm[INDEX_MCP].x
    ):
        return "S"
    if (
        idx_curl and mid_curl and ring_curl and pinky_curl
        and landmark_distance(lm[THUMB_TIP], lm[INDEX_PIP]) < 0.05
    ):
        return "T"
    if (
        idx_up and mid_up and not ring_up and not pinky_up
        and landmark_distance(lm[INDEX_TIP], lm[MIDDLE_TIP]) < 0.04
    ):
        return "U"
    if (
        idx_up and mid_up and not ring_up and not pinky_up
        and landmark_distance(lm[INDEX_TIP], lm[MIDDLE_TIP]) > 0.05
    ):
        return "V"
    if idx_up and mid_up and ring_up and not pinky_up and not thumb:
        return "W"
    if (
        not idx_up and not mid_up and not ring_up and not pinky_up
        and lm[INDEX_TIP].y > lm[INDEX_PIP].y
        and landmark_distance(lm[INDEX_TIP], lm[INDEX_MCP]) < 0.07
    ):
        return "X"
    if thumb and not idx_up and not mid_up and not ring_up and pinky_up:
        return "Y"
    if idx_up and not mid_up and not ring_up and not pinky_up and not thumb:
        return "Z"

    return None


def _recognize_word_gesture(lm) -> tuple[Optional[str], Optional[str]]:
    idx_up = is_finger_up(lm, INDEX_TIP, INDEX_PIP)
    mid_up = is_finger_up(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring_up = is_finger_up(lm, RING_TIP, RING_PIP)
    pinky_up = is_finger_up(lm, PINKY_TIP, PINKY_PIP)
    thumb = is_thumb_up(lm)

    if thumb and idx_up and mid_up and ring_up and pinky_up:
        return "Halo!", "Halo"
    if is_fist(lm) and not thumb:
        return "Nama Saya", "Nama saya"
    if thumb and idx_up and not mid_up and not ring_up and pinky_up:
        return "Salsa", "Salsa"
    if thumb and not idx_up and not mid_up and not ring_up and not pinky_up:
        return "Bagus!", "Bagus"
    if (
        idx_up and mid_up and not ring_up and not pinky_up
        and landmark_distance(lm[INDEX_TIP], lm[MIDDLE_TIP]) > 0.05
        and not thumb
    ):
        return "Damai", "Damai"
    if idx_up and not mid_up and not ring_up and pinky_up and not thumb:
        return "Asyik!", "Asyik"

    return None, None


def recognize(landmarks, ml_model=None, label_encoder=None) -> RecognitionResult:
    word_label, word_speech = _recognize_word_gesture(landmarks)
    if word_label:
        return word_label, word_speech, (0, 200, 255), "KATA"

    if ml_model is not None:
        ml_label, confidence = _recognize_ml(landmarks, ml_model, label_encoder)
        if confidence > ML_CONFIDENCE_THRESHOLD:
            display = f"BISINDO: {ml_label} ({confidence * 100:.0f}%)"
            return display, ml_label, (0, 255, 100), "ML"

    rule_label = _recognize_bisindo_rules(landmarks)
    if rule_label:
        return f"BISINDO: {rule_label}", rule_label, (100, 255, 200), "RULE"

    return "...", None, (200, 200, 200), ""
