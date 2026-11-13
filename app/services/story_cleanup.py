"""Removes a finished story's chunks and final files from the server disk once its compressed copy
is safely in the bucket and nobody is on it any more. A story left halfway has its chunks packed
into the bucket instead (story_partial.py), and restored when it is played again."""

import asyncio
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.services import storage, story_audio_store, story_partial
from app.services.story_stream import story_busy, story_in_use, story_on_device
from app.store import TTS_DIR

# A phone still on the story keeps its files until it starts another one (then they go at once),
# or until it has not played it for a day (app closed for good).
KEEP_ON_DEVICE_S = 86400
# Nobody on it (the phone moved on, or the server restarted): save_position updates last_played_at
# every 15–30 s while it plays, so this long without that means nobody is listening.
KEEP_AFTER_PLAY_S = 900
# A story left halfway stays on disk this long after it was last played, then it is packed.
PACK_AFTER_S = 1800
CLEANUP_EVERY_S = 300


def _streams_dir() -> Path:
    return TTS_DIR / "story_streams"


async def remove_story_files(story_key: str, final_key: str | None, folder: Path) -> bool:
    """Deletes the local folder once the bucket really holds the final file."""
    # A local final_key means the upload failed: the disk copy is the only one.
    if not storage.is_stored(final_key):
        return False
    try:
        if not await asyncio.to_thread(storage.exists, final_key):
            print(f"[cleanup] {story_key[:8]}: {final_key} missing in the bucket, kept local", flush=True)
            return False
    except Exception as error:
        print(f"[cleanup] {story_key[:8]}: bucket check failed, kept local: {error}", flush=True)
        return False
    await asyncio.to_thread(shutil.rmtree, folder, True)
    print(f"[cleanup] {story_key[:8]}: local files removed, plays from {final_key}", flush=True)
    return True


async def remove_left_story(story_key: str) -> bool:
    """The phone on this story started another one: remove its files now if they are in the bucket.
    Not finished or not uploaded yet: the sweep handles it later."""
    folder = _streams_dir() / story_key
    if not storage.storage_configured() or not folder.is_dir() or story_in_use(story_key):
        return False
    try:
        rows = await story_audio_store.ready_rows([story_key])
        return bool(rows) and await remove_story_files(story_key, rows[0][1], folder)
    except Exception as error:
        print(f"[cleanup] {story_key[:8]}: removal failed, the sweep retries: {error}", flush=True)
        return False


async def clean_story_files() -> int:
    """One sweep. Returns how many story folders were removed (finished or packed)."""
    streams_dir = _streams_dir()
    if not storage.storage_configured() or not streams_dir.is_dir():
        return 0
    folders = {path.name: path for path in streams_dir.iterdir() if path.is_dir()}
    rows = await story_audio_store.ready_rows(list(folders))
    now = datetime.now(timezone.utc)
    removed = 0
    for story_key, final_key, last_used in rows:
        keep_s = KEEP_ON_DEVICE_S if story_on_device(story_key) else KEEP_AFTER_PLAY_S
        if story_busy(story_key) or last_used > now - timedelta(seconds=keep_s):
            continue
        removed += await remove_story_files(story_key, final_key, folders[story_key])
    # Left halfway or failed: packed into the bucket, restored when played again.
    for story_key, last_used in await story_audio_store.partial_rows(list(folders)):
        keep_s = KEEP_ON_DEVICE_S if story_on_device(story_key) else PACK_AFTER_S
        if story_busy(story_key) or last_used > now - timedelta(seconds=keep_s):
            continue
        removed += await story_partial.pack_left_story(story_key, folders[story_key], story_busy)
    return removed


async def watch_story_files() -> None:
    while True:
        try:
            await clean_story_files()
        except Exception as error:
            # A database or bucket problem only skips this sweep.
            print(f"[cleanup] sweep failed: {error}", flush=True)
        await asyncio.sleep(CLEANUP_EVERY_S)
