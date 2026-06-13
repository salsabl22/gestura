import csv
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src.config import (
    MODEL_PATH,
    DATASET_FILE,
    SAMPLES_PER_LETTER,
    LETTERS,
    HAND_DETECTION_CONFIDENCE,
    HAND_PRESENCE_CONFIDENCE,
    HAND_TRACKING_CONFIDENCE,
)
from src.model_downloader import ensure_model_exists
from src.landmarks import HAND_CONNECTIONS

WINDOW_TITLE = "BISINDO Dataset Collector"
COUNTDOWN_SECONDS = 3


def init_dataset_file():
    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
    if not os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
            writer.writerow(header)
        print(f"Dataset file created: {DATASET_FILE}")


def count_samples() -> dict:
    counts = {letter: 0 for letter in LETTERS}
    if not os.path.exists(DATASET_FILE):
        return counts
    with open(DATASET_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row and row[0] in counts:
                counts[row[0]] += 1
    return counts


def save_landmark(label: str, landmarks: list):
    with open(DATASET_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        row = [label] + [coord for lm in landmarks for coord in (round(lm.x, 6), round(lm.y, 6), round(lm.z, 6))]
        writer.writerow(row)


def draw_landmarks(image, landmarks):
    h, w = image.shape[:2]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
    for start, end in HAND_CONNECTIONS:
        x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
        x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)


def draw_ui(frame, current_letter: str, current_idx: int, counts: dict, collecting: bool, countdown_remaining: int):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"Huruf: {current_letter}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    current_count = counts[current_letter]
    progress = min(current_count / SAMPLES_PER_LETTER, 1.0)
    bar_w = w - 40
    cv2.rectangle(frame, (20, 65), (20 + bar_w, 90), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 65), (20 + int(bar_w * progress), 90), (0, 255, 100), -1)
    cv2.putText(frame, f"{current_count}/{SAMPLES_PER_LETTER}", (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if collecting and countdown_remaining == 0:
        cv2.putText(frame, "● RECORDING", (w - 200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    panel_y = h - 90
    cv2.rectangle(frame, (0, panel_y), (w, h), (0, 0, 0), -1)
    for i, letter in enumerate(LETTERS):
        done = counts[letter] >= SAMPLES_PER_LETTER
        color = (0, 255, 255) if i == current_idx else ((0, 255, 0) if done else (100, 100, 100))
        x_offset = 10 + i * 28
        cv2.putText(frame, letter, (x_offset, panel_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        dot_color = (0, 200, 0) if done else (60, 60, 60)
        cv2.circle(frame, (x_offset + 7, panel_y + 50), 4, dot_color, -1)

    cv2.putText(frame, "SPACE=Record  N=Next  P=Prev  Q=Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def print_summary(counts: dict):
    print("\n=== DATASET SUMMARY ===")
    total = 0
    for letter, count in counts.items():
        status = "OK" if count >= SAMPLES_PER_LETTER else f"{count}/{SAMPLES_PER_LETTER}"
        print(f"  {letter}: {status}")
        total += count
    print(f"\nTotal samples: {total}")
    print(f"File: {DATASET_FILE}")


def main():
    ensure_model_exists()
    init_dataset_file()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera.")
            return

        current_idx = 0
        collecting = False
        countdown_start = 0.0
        countdown_remaining = 0

        print("\n=== BISINDO DATASET COLLECTOR ===")
        print("SPACE = start collecting current letter")
        print("N     = next letter")
        print("P     = previous letter")
        print("Q     = quit\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            counts = count_samples()
            current_letter = LETTERS[current_idx]
            current_count = counts[current_letter]

            if collecting and result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                elapsed = time.time() - countdown_start
                countdown_remaining = max(0, COUNTDOWN_SECONDS - int(elapsed))

                if countdown_remaining > 0:
                    cv2.putText(frame, f"Ready in {countdown_remaining}...",
                                (w // 2 - 120, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                else:
                    if current_count < SAMPLES_PER_LETTER:
                        save_landmark(current_letter, landmarks)
                    else:
                        collecting = False
                        print(f"Done: Letter {current_letter} ({SAMPLES_PER_LETTER} samples)")

            if result.hand_landmarks:
                draw_landmarks(frame, result.hand_landmarks[0])

            draw_ui(frame, current_letter, current_idx, counts, collecting, countdown_remaining)
            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                if current_count < SAMPLES_PER_LETTER:
                    collecting = True
                    countdown_start = time.time()
                    countdown_remaining = COUNTDOWN_SECONDS
                    print(f"Collecting letter: {current_letter}...")
                else:
                    print(f"Letter {current_letter} already complete.")
            elif key == ord("n"):
                current_idx = (current_idx + 1) % len(LETTERS)
                collecting = False
            elif key == ord("p"):
                current_idx = (current_idx - 1) % len(LETTERS)
                collecting = False
            elif cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()
        print_summary(count_samples())


if __name__ == "__main__":
    main()
