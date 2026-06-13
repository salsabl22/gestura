import queue
import threading
import time
from typing import Optional

from src.config import SPEAK_COOLDOWN, SPEAK_REPEAT_INTERVAL, TTS_SPEECH_RATE, TTS_VOLUME


class TTSEngine:
    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._last_spoken: str = ""
        self._last_speak_time: float = 0.0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_SPEECH_RATE)
        engine.setProperty("volume", TTS_VOLUME)

        for voice in engine.getProperty("voices"):
            if "indonesia" in voice.name.lower() or "id" in voice.id.lower():
                engine.setProperty("voice", voice.id)
                break

        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS] Error: {e}")
            self._queue.task_done()

    def speak(self, text: str):
        now = time.time()
        gesture_changed = text != self._last_spoken
        cooldown_passed = (now - self._last_speak_time) > SPEAK_COOLDOWN
        repeat_due = (now - self._last_speak_time) > SPEAK_REPEAT_INTERVAL

        if cooldown_passed and (gesture_changed or repeat_due):
            self._flush_queue()
            self._queue.put(text)
            self._last_spoken = text
            self._last_speak_time = now

    def _flush_queue(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def is_speaking(self) -> bool:
        return not self._queue.empty()

    def stop(self):
        self._queue.put(None)
