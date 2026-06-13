import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src.config import (
    MODEL_PATH,
    MAX_HANDS,
    HAND_DETECTION_CONFIDENCE,
    HAND_PRESENCE_CONFIDENCE,
    HAND_TRACKING_CONFIDENCE,
)
from src.model_downloader import ensure_model_exists
from src.recognizer import recognize, load_ml_model
from src.tts_engine import TTSEngine
from src.drawing import (
    draw_hand_landmarks,
    draw_text_with_background,
    draw_method_badge,
    draw_speaking_indicator,
    draw_exit_hint,
)

WINDOW_TITLE = "BISINDO Hand Recognition"


def process_frame(image, detection_result, tts: TTSEngine, ml_model, label_encoder):
    spoken = []

    for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
        label, speech_text, color, method = recognize(hand_landmarks, ml_model, label_encoder)
        draw_hand_landmarks(image, hand_landmarks)

        if method:
            draw_method_badge(image, method, i)

        y_pos = 60 + i * 55
        draw_text_with_background(image, label, (20, y_pos), font_scale=1.2, color=color)

        if speech_text and speech_text not in spoken:
            spoken.append(speech_text)

    if spoken:
        tts.speak(", ".join(spoken))

    if tts.is_speaking():
        draw_speaking_indicator(image)

    draw_exit_hint(image)

    return image


def should_exit(key: int, window_title: str) -> bool:
    if key == ord("q") or key == 27:
        return True
    if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
        return True
    return False


def main():
    ensure_model_exists()

    ml_model, label_encoder = load_ml_model()
    if ml_model:
        print(f"ML model loaded from: {__import__('src.config', fromlist=['ML_MODEL_PATH']).ML_MODEL_PATH}")
    else:
        print("ML model not found. Using rule-based recognition only.")

    tts = TTSEngine()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera.")
            return

        print("\nCamera active. Press Q or ESC to quit.\n")
        print("Supported word gestures:")
        print("  Halo       — all fingers open")
        print("  Nama Saya  — closed fist")
        print("  Salsa      — thumb + index + pinky")
        print("  Bagus      — thumb only")
        print("  Damai      — index + middle (V, spread)")
        print("  Asyik      — index + pinky\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            frame = process_frame(frame, result, tts, ml_model, label_encoder)
            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if should_exit(key, WINDOW_TITLE):
                break

        cap.release()
        cv2.destroyAllWindows()
        tts.stop()
        print("Program finished.")


if __name__ == "__main__":
    main()
