import hashlib
import re
from pathlib import Path

import edge_tts

from app.services.audio import (
    audio_duration,
    concat_wav_bytes,
    delete_file,
    ensure_wav,
    find_cloned_tts_path,
    find_tts_path,
    tts_path,
)
from app.services.chatterbox_client import colab_configured, generate_on_colab
from app.store import load_voices

CHUNK_CHARS = 320

VOICE_BY_LANGUAGE = {
    "hi": "hi-IN-SwaraNeural",
    "hi-in": "hi-IN-SwaraNeural",
    "hindi": "hi-IN-SwaraNeural",
    "en": "en-IN-NeerjaNeural",
    "en-in": "en-IN-NeerjaNeural",
    "english": "en-IN-NeerjaNeural",
    "hinglish": "en-IN-NeerjaNeural",
}


class TtsUnavailable(Exception):
    pass


def normalize_language(language: str | None) -> str:
    value = (language or "english").strip().lower()
    if "hindi" in value or value in {"hi", "hi-in"}:
        return "hindi"
    return "english"


def language_id(language: str | None) -> str:
    return "hi" if normalize_language(language) == "hindi" else "en"


def cache_id(text: str, language: str, voice_id: str | None = None) -> str:
    key = f"{voice_id or ''}|{normalize_language(language)}|{text.strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def split_sentences(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.?!।])\s+", cleaned)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        candidate = f"{buf} {piece}".strip() if buf else piece
        if len(candidate) <= CHUNK_CHARS:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        if len(piece) <= CHUNK_CHARS:
            buf = piece
            continue
        for start in range(0, len(piece), CHUNK_CHARS):
            slice_text = piece[start : start + CHUNK_CHARS].strip()
            if slice_text:
                chunks.append(slice_text)
        buf = ""
    if buf:
        chunks.append(buf)
    return chunks or [cleaned]


def _readable(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists() and path.stat().st_size > 0:
        return path
    return None


def resolve_prompt(voice_id: str | None) -> tuple[str, Path] | None:
    voices = load_voices()
    ordered: list[dict] = []
    if voice_id:
        match = next((item for item in voices if item.get("id") == voice_id), None)
        if match:
            ordered.append(match)
    ordered.extend(item for item in voices if item.get("is_default") and item not in ordered)
    ordered.extend(item for item in voices if item not in ordered)

    for item in ordered:
        source = _readable(item.get("file_path"))
        if source is None:
            continue
        prompt = ensure_wav(source)
        if prompt.exists() and prompt.stat().st_size > 0:
            return str(item.get("id") or ""), prompt
    return None


async def _edge_tts(text: str, language: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.mp3")
    try:
        voice = VOICE_BY_LANGUAGE[normalize_language(language)]
        communicate = edge_tts.Communicate(text.strip(), voice)
        await communicate.save(str(tmp))
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError("edge-tts produced empty audio")
        tmp.replace(dest)
        return dest
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


async def _clone_on_colab(text: str, language: str, prompt_wav: Path, dest: Path) -> Path:
    lang = language_id(language)
    chunks = split_sentences(text)
    wavs: list[bytes] = []
    for chunk in chunks:
        audio = await generate_on_colab(chunk, lang, str(prompt_wav))
        if not audio:
            raise RuntimeError("Colab produced empty audio")
        wavs.append(audio)
    if len(wavs) == 1:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp.wav")
        tmp.write_bytes(wavs[0])
        tmp.replace(dest)
        return dest
    return concat_wav_bytes(wavs, dest)


async def synthesize(
    text: str,
    language: str,
    voice_id: str | None = None,
) -> tuple[str, Path, float, bool]:
    prompt = resolve_prompt(voice_id)
    prompt_id = prompt[0] if prompt else (voice_id or "")
    tts_id = cache_id(text, language, prompt_id)
    want_clone = colab_configured() and bool(prompt)
    cloned_cache = find_cloned_tts_path(tts_id) if want_clone else None
    if cloned_cache:
        return tts_id, cloned_cache, audio_duration(cloned_cache), True

    if want_clone:
        wav_dest = tts_path(tts_id, "wav")
        try:
            path = await _clone_on_colab(text.strip(), language, prompt[1], wav_dest)
            delete_file(str(tts_path(tts_id, "mp3")))
            return tts_id, path, audio_duration(path), True
        except Exception as error:
            delete_file(str(wav_dest.with_suffix(".tmp.wav")))
            delete_file(str(wav_dest))
            print(f"[tts] colab clone failed fallback edge: {error}", flush=True)
    else:
        cached = find_tts_path(tts_id)
        if cached:
            return tts_id, cached, audio_duration(cached), cached.suffix.lower() == ".wav"

    mp3_dest = tts_path(tts_id, "mp3")
    try:
        path = await _edge_tts(text, language, mp3_dest)
        return tts_id, path, audio_duration(path), False
    except Exception as error:
        delete_file(str(mp3_dest))
        if want_clone:
            raise TtsUnavailable(
                "Clone unavailable (Colab down) and Edge TTS failed"
            ) from error
        raise TtsUnavailable("Edge TTS failed") from error
