from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"
TTS_DIR = DATA_DIR / "tts"

# Voices live in the PostgreSQL "voices" table (app/services/voice_store.py); only audio files are here.


def ensure_dirs() -> None:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    TTS_DIR.mkdir(parents=True, exist_ok=True)
