import json
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VOICES_PATH = DATA_DIR / "voices.json"
RECORDINGS_DIR = DATA_DIR / "recordings"
TTS_DIR = DATA_DIR / "tts"

_lock = Lock()


def ensure_dirs() -> None:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    if not VOICES_PATH.exists():
        VOICES_PATH.write_text("[]", encoding="utf-8")


def load_voices() -> list[dict]:
    ensure_dirs()
    with _lock:
        return json.loads(VOICES_PATH.read_text(encoding="utf-8"))


def save_voices(voices: list[dict]) -> None:
    ensure_dirs()
    with _lock:
        VOICES_PATH.write_text(json.dumps(voices, indent=2), encoding="utf-8")
