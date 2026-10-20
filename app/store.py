import json
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VOICES_PATH = DATA_DIR / "voices.json"
RECORDINGS_DIR = DATA_DIR / "recordings"
TTS_DIR = DATA_DIR / "tts"

_lock = Lock()
# voices.json text, keyed by (mtime, size): read from disk only when the file changed.
_voices_cache: tuple[tuple[int, int], str] | None = None


def ensure_dirs() -> None:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    if not VOICES_PATH.exists():
        VOICES_PATH.write_text("[]", encoding="utf-8")


def load_voices() -> list[dict]:
    ensure_dirs()
    global _voices_cache
    with _lock:
        stat = VOICES_PATH.stat()
        version = (stat.st_mtime_ns, stat.st_size)
        if _voices_cache is None or _voices_cache[0] != version:
            _voices_cache = (version, VOICES_PATH.read_text(encoding="utf-8"))
        # Parsed fresh each call: callers change the list before save_voices.
        return json.loads(_voices_cache[1])


def save_voices(voices: list[dict]) -> None:
    ensure_dirs()
    with _lock:
        VOICES_PATH.write_text(json.dumps(voices, indent=2), encoding="utf-8")
