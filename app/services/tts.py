import hashlib
from pathlib import Path

import edge_tts
from mutagen.mp3 import MP3

from app.services.audio import tts_path

VOICE_BY_LANGUAGE = {
    "hi": "hi-IN-SwaraNeural",
    "hi-in": "hi-IN-SwaraNeural",
    "hindi": "hi-IN-SwaraNeural",
    "en": "en-IN-NeerjaNeural",
    "en-in": "en-IN-NeerjaNeural",
    "english": "en-IN-NeerjaNeural",
    "hinglish": "en-IN-NeerjaNeural",
}


def normalize_language(language: str | None) -> str:
    value = (language or "english").strip().lower()
    if "hindi" in value or value in {"hi", "hi-in"}:
        return "hindi"
    return "english"


def cache_id(text: str, language: str) -> str:
    key = f"{normalize_language(language)}|{text.strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def mp3_duration(path: Path) -> float:
    try:
        return max(1.0, float(MP3(path).info.length))
    except Exception:
        return 1.0


async def synthesize(text: str, language: str) -> tuple[str, Path, float]:
    tts_id = cache_id(text, language)
    path = tts_path(tts_id)
    if path.exists() and path.stat().st_size > 0:
        return tts_id, path, mp3_duration(path)

    voice = VOICE_BY_LANGUAGE[normalize_language(language)]
    communicate = edge_tts.Communicate(text.strip(), voice)
    await communicate.save(str(path))
    return tts_id, path, mp3_duration(path)
