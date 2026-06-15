# Gestura

<img width="400" alt="image (3)" src="https://github.com/user-attachments/assets/20d9e0a0-a60f-4879-affe-6026f66b3695" />
<img width="400" alt="image (4)" src="https://github.com/user-attachments/assets/9bbde065-8589-4fda-882c-e092fc928e4b" />



Real-time BISINDO (Indonesian Sign Language) hand sign detection using MediaPipe Hands, OpenCV, and scikit-learn. Supports both rule-based and ML-based recognition with text-to-speech output.

---

## Features

- Real-time hand landmark detection via MediaPipe
- Rule-based recognition for all 26 BISINDO alphabet letters (A–Z)
- ML-based recognition using a trained Random Forest classifier
- Custom word gesture recognition (e.g., "Halo", "Nama Saya", "Salsa")
- Text-to-speech output using pyttsx3 with Indonesian voice support
- Dataset collection tool for training custom models

---

## Project Structure

```
gestura/
├── main.py                     # Entry point: real-time recognition
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── config.py               # All constants and configuration
│   ├── landmarks.py            # MediaPipe landmark index constants
│   ├── gesture_utils.py        # Low-level gesture helper functions
│   ├── recognizer.py           # Recognition logic (rules + ML + word gestures)
│   ├── tts_engine.py           # Thread-safe TTS engine
│   ├── drawing.py              # OpenCV drawing utilities
│   └── model_downloader.py     # Auto-download MediaPipe model
│
├── scripts/
│   ├── collect_dataset.py      # Interactive dataset collection tool
│   └── train_model.py          # Model training script
│
├── data/                       # bisindo_dataset.csv (generated, git-ignored)
└── models/                     # hand_landmarker.task, bisindo_model.pkl (git-ignored)
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run real-time recognition

```bash
python main.py
```

The MediaPipe hand landmarker model will be downloaded automatically on first run.

### Collect training data (optional)

```bash
python scripts/collect_dataset.py
```

Controls:
- `SPACE` — start recording for the current letter
- `N` — next letter
- `P` — previous letter
- `Q` / `ESC` — quit

### Train the ML model (optional)

```bash
python scripts/train_model.py
```

Requires `data/bisindo_dataset.csv` to exist. The trained model is saved to `models/bisindo_model.pkl` and will be loaded automatically by `main.py`.

---

## Recognition Modes

| Mode | Description |
|------|-------------|
| `KATA` | Custom word gestures (highest priority) |
| `ML` | ML-based prediction with confidence > 75% |
| `RULE` | Rule-based fallback for A–Z alphabet |

---

## Supported Word Gestures

| Gesture | Hand Shape |
|---------|-----------|
| Halo | All five fingers open |
| Nama Saya | Closed fist |
| Salsa | Thumb + index + pinky |
| Bagus | Thumb only |
| Damai | Index + middle spread (V) |
| Asik | Index + pinky |

---

## Controls (during recognition)

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `ESC` | Quit |

---

## Requirements

- Python 3.10+
- Webcam

---
