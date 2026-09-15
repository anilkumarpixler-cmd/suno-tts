from pathlib import Path

from fastapi import UploadFile

from app.store import RECORDINGS_DIR, TTS_DIR, ensure_dirs


def _extension(filename: str | None) -> str:
    if not filename or "." not in filename:
        return "m4a"
    return filename.rsplit(".", 1)[-1].lower()[:8]


def save_bytes(voice_id: str, data: bytes, filename: str | None) -> Path:
    ensure_dirs()
    dest = RECORDINGS_DIR / f"{voice_id}.{_extension(filename)}"
    dest.write_bytes(data)
    return dest


async def save_recording(voice_id: str, upload: UploadFile) -> Path:
    return save_bytes(voice_id, await upload.read(), upload.filename)


def delete_file(path: str | None) -> None:
    if not path:
        return
    file = Path(path)
    if file.exists():
        file.unlink()


def tts_path(tts_id: str) -> Path:
    ensure_dirs()
    return TTS_DIR / f"{tts_id}.mp3"
