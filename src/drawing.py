import cv2
import numpy as np

from src.landmarks import HAND_CONNECTIONS, FINGERTIP_INDICES

METHOD_BADGE_COLORS = {
    "KATA": (0, 165, 255),
    "ML": (0, 200, 50),
    "RULE": (180, 180, 0),
}


def draw_hand_landmarks(image: np.ndarray, landmarks: list):
    h, w = image.shape[:2]
    for i, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        color = (0, 0, 255) if i in FINGERTIP_INDICES else (0, 255, 0)
        cv2.circle(image, (cx, cy), 6, color, -1)
        cv2.circle(image, (cx, cy), 6, (255, 255, 255), 1)
    for start, end in HAND_CONNECTIONS:
        x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
        x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)


def draw_text_with_background(
    image: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_scale: float = 1.0,
    color: tuple = (255, 255, 255),
    thickness: int = 2,
):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(image, (x - 5, y - th - 8), (x + tw + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_method_badge(image: np.ndarray, method: str, hand_index: int):
    h, w = image.shape[:2]
    color = METHOD_BADGE_COLORS.get(method, (100, 100, 100))
    cv2.rectangle(image, (w - 110, 5 + hand_index * 30), (w - 5, 28 + hand_index * 30), color, -1)
    cv2.putText(image, method, (w - 102, 22 + hand_index * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def draw_speaking_indicator(image: np.ndarray):
    h, w = image.shape[:2]
    cv2.circle(image, (w - 20, h - 20), 10, (0, 255, 255), -1)
    cv2.putText(image, chr(9834), (w - 27, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_exit_hint(image: np.ndarray):
    h, w = image.shape[:2]
    cv2.putText(image, "Q / ESC = keluar",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
