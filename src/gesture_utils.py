import math
from src.landmarks import (
    THUMB_TIP, THUMB_IP,
    INDEX_TIP, INDEX_PIP, INDEX_MCP,
    MIDDLE_TIP, MIDDLE_MCP,
    RING_TIP, RING_MCP,
    PINKY_TIP, PINKY_MCP,
)


def landmark_distance(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def is_finger_up(landmarks, tip: int, pip: int) -> bool:
    return landmarks[tip].y < landmarks[pip].y


def is_finger_curled(landmarks, tip: int, mcp: int) -> bool:
    return landmarks[tip].y > landmarks[mcp].y


def is_thumb_up(landmarks) -> bool:
    return landmarks[THUMB_TIP].y < landmarks[THUMB_IP].y


def are_fingers_touching(landmarks, a: int, b: int, threshold: float = 0.05) -> bool:
    return landmark_distance(landmarks[a], landmarks[b]) < threshold


def is_fist(landmarks) -> bool:
    return (
        is_finger_curled(landmarks, INDEX_TIP, INDEX_MCP)
        and is_finger_curled(landmarks, MIDDLE_TIP, MIDDLE_MCP)
        and is_finger_curled(landmarks, RING_TIP, RING_MCP)
        and is_finger_curled(landmarks, PINKY_TIP, PINKY_MCP)
    )
