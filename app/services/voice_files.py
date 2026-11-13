"""A voice's files. On the server disk (data/recordings/): the original recording `<id>.<ext>`, its
cleaned prompt `<id>.prompt.wav` and the GPU's prepared voice `<id>.prompt.wav.<key>.conds.pt`.
The original and the prompt are also copied to the bucket under voices/<id>/, so a lost disk loses
no voice. The conds file stays local only: it is a cache the GPU rebuilds from the prompt.
Calls block: run them in a thread."""

from pathlib import Path

from app.services import storage
from app.store import RECORDINGS_DIR


def bucket_key(voice_id: str, path: Path) -> str:
    name = "prompt.wav" if path.name.endswith(".prompt.wav") else f"original{path.suffix.lower()}"
    return f"{storage.VOICE_PREFIX}{voice_id}/{name}"


def back_up(voice_id: str, paths: list[Path]) -> None:
    if not storage.storage_configured():
        return
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            storage.upload_file(path, bucket_key(voice_id, path))


def restore(voice_id: str, path: Path) -> Path | None:
    """The local file, downloaded from the bucket first when the disk lost it. None if neither has it."""
    if path.exists() and path.stat().st_size > 0:
        return path
    if not storage.storage_configured():
        return None
    try:
        storage.download_file(bucket_key(voice_id, path), path)
    except Exception as error:
        print(f"[voice] {voice_id}: {path.name} not restored from the bucket: {error}", flush=True)
        return None
    print(f"[voice] {voice_id}: restored {path.name} from the bucket", flush=True)
    return path


def remove(voice_id: str) -> None:
    # Every local file of the voice starts with "<id>." (ids are "v_<10 hex>", so no other voice matches).
    for path in RECORDINGS_DIR.glob(f"{voice_id}.*"):
        path.unlink(missing_ok=True)
    if storage.storage_configured():
        storage.delete_prefix(f"{storage.VOICE_PREFIX}{voice_id}/")
