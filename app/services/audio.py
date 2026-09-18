from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import wave

from fastapi import UploadFile
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

from app.store import RECORDINGS_DIR, TTS_DIR, ensure_dirs

WAV_EXTENSIONS = {".wav"}


def _extension(filename: str | None) -> str:
    if not filename or "." not in filename:
        return "m4a"
    return filename.rsplit(".", 1)[-1].lower()[:8]


def save_bytes(voice_id: str, data: bytes, filename: str | None) -> Path:
    ensure_dirs()
    dest = RECORDINGS_DIR / f"{voice_id}.{_extension(filename)}"
    dest.write_bytes(data)
    wav = ensure_wav(dest)
    return wav if wav.suffix.lower() == ".wav" else dest


async def save_recording(voice_id: str, upload: UploadFile) -> Path:
    return save_bytes(voice_id, await upload.read(), upload.filename)


def delete_file(path: str | None) -> None:
    if not path:
        return
    file = Path(path)
    if file.exists():
        file.unlink()


def tts_path(tts_id: str, ext: str = "mp3") -> Path:
    ensure_dirs()
    suffix = ext if ext.startswith(".") else f".{ext}"
    return TTS_DIR / f"{tts_id}{suffix}"


def find_tts_path(tts_id: str) -> Path | None:
    ensure_dirs()
    for suffix in (".wav", ".mp3"):
        path = TTS_DIR / f"{tts_id}{suffix}"
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def find_cloned_tts_path(tts_id: str) -> Path | None:
    ensure_dirs()
    path = TTS_DIR / f"{tts_id}.wav"
    if path.exists() and path.stat().st_size > 0:
        return path
    return None


def concat_wav_bytes(parts: list[bytes], dest: Path) -> Path:
    if not parts:
        raise RuntimeError("No wav chunks to stitch")
    frames: list[bytes] = []
    params = None
    for data in parts:
        with wave.open(BytesIO(data), "rb") as src:
            next_params = src.getparams()
            if params is None:
                params = next_params
            elif (src.getnchannels(), src.getsampwidth(), src.getframerate()) != (
                params.nchannels,
                params.sampwidth,
                params.framerate,
            ):
                raise RuntimeError("Colab wav chunks have mismatched format")
            frames.append(src.readframes(src.getnframes()))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.wav")
    with wave.open(str(tmp), "wb") as out:
        assert params is not None
        out.setparams(params)
        for chunk in frames:
            out.writeframes(chunk)
    tmp.replace(dest)
    return dest


def media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix in {".m4a", ".mp4"}:
        return "audio/mp4"
    return "audio/mpeg"


def audio_duration(path: Path) -> float:
    try:
        if path.suffix.lower() == ".wav":
            return max(1.0, float(WAVE(path).info.length))
        return max(1.0, float(MP3(path).info.length))
    except Exception:
        return 1.0


def _ffmpeg_bin() -> str | None:
    for name in ("ffmpeg", "ffmpeg.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


def ensure_wav(path: Path) -> Path:
    if not path.exists():
        return path
    if path.suffix.lower() in WAV_EXTENSIONS:
        return path

    wav = path.with_suffix(".wav")
    if wav.exists() and wav.stat().st_size > 0:
        return wav

    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        return path

    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-ac", "1", "-ar", "24000", str(wav)],
            check=True,
            capture_output=True,
        )
        if wav.exists() and wav.stat().st_size > 0:
            return wav
    except Exception:
        wav.unlink(missing_ok=True)
    return path
