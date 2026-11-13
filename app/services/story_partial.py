"""A story left halfway: once nobody is on it for a while, its made chunks are joined into one
lossless tmp/<key>/partial.flac in the bucket and the local folder is removed. Played again, the
FLAC is split back into chunk_N.wav before anything is generated, so streaming continues from the
first missing chunk. pack/unpack block: run them in a thread."""

import asyncio
import contextlib
import shutil
import subprocess
import wave
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from app.services import storage, story_audio_store
from app.services.audio import STORY_FLAC_ARGS, _ffmpeg_bin, _ffmpeg_encode, concat_wav_bytes

PARTIAL_NAME = "partial.flac"

# story_key -> (lock, how many callers hold or wait on it), so a pack and a restore of the same
# story never overlap. Dropped once nobody uses it.
_locks: dict[str, tuple[asyncio.Lock, int]] = {}


@contextlib.asynccontextmanager
async def _story_lock(story_key: str):
    lock, users = _locks.get(story_key, (None, 0))
    lock = lock or asyncio.Lock()
    _locks[story_key] = (lock, users + 1)
    try:
        async with lock:
            yield
    finally:
        lock, users = _locks[story_key]
        if users <= 1:
            del _locks[story_key]
        else:
            _locks[story_key] = (lock, users - 1)


def _chunk_path(folder: Path, index: int) -> Path:
    return folder / f"chunk_{index}.wav"


def _made_chunks(folder: Path) -> list[Path]:
    # chunk_0 onwards; a gap ends the run (the chunks after it are made again).
    paths: list[Path] = []
    while (path := _chunk_path(folder, len(paths))).exists() and path.stat().st_size > 0:
        paths.append(path)
    return paths


def pack(folder: Path) -> tuple[Path, list[int]] | None:
    """(partial.flac in `folder`, frames of each chunk), or None if there is no chunk 0, a chunk is
    not 16-bit, or ffmpeg is missing."""
    paths = _made_chunks(folder)
    if not paths:
        return None
    parts = [path.read_bytes() for path in paths]
    samples: list[int] = []
    for data in parts:
        with wave.open(BytesIO(data), "rb") as src:
            # unpack decodes back to 16-bit PCM.
            if src.getsampwidth() != 2:
                return None
            samples.append(src.getnframes())
    joined = concat_wav_bytes(parts, folder / "partial.wav")
    try:
        flac = _ffmpeg_encode(joined, folder / PARTIAL_NAME, STORY_FLAC_ARGS)
    finally:
        joined.unlink(missing_ok=True)
    return (flac, samples) if flac else None


def unpack(flac: Path, samples: list[int], folder: Path) -> int:
    """Splits `flac` back into chunk_N.wav (`samples` frames each). Chunks already on disk are kept.
    Returns how many were written."""
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg missing")
    joined = folder / "partial.wav"
    written = 0
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(flac), "-c:a", "pcm_s16le", str(joined)],
            check=True,
            capture_output=True,
        )
        with wave.open(str(joined), "rb") as src:
            if src.getnframes() != sum(samples):
                raise RuntimeError(f"{src.getnframes()} frames decoded, {sum(samples)} expected")
            params = src.getparams()
            for index, frames in enumerate(samples):
                data = src.readframes(frames)
                path = _chunk_path(folder, index)
                if path.exists() and path.stat().st_size > 0:
                    continue
                out = BytesIO()
                with wave.open(out, "wb") as dst:
                    dst.setparams(params)
                    dst.writeframes(data)
                # Through a temp file, so a reader never sees half a chunk.
                tmp = path.with_name(f"{path.name}.tmp")
                tmp.write_bytes(out.getvalue())
                tmp.replace(path)
                written += 1
    finally:
        joined.unlink(missing_ok=True)
    return written


async def delete_partial(story_key: str, key: str) -> None:
    # Best effort: the bucket lifecycle rule deletes it anyway.
    try:
        await asyncio.to_thread(storage.delete_file, key)
    except Exception as error:
        print(f"[partial] {story_key[:8]}: could not delete {key}: {error}", flush=True)


async def pack_left_story(story_key: str, folder: Path, busy: Callable[[str], bool]) -> bool:
    """Packs the story's chunks into the bucket and removes `folder`. False (folder kept) when there
    is nothing to pack, the upload or database write failed, or the story is busy again."""
    async with _story_lock(story_key):
        if busy(story_key) or not folder.is_dir():
            return False
        try:
            packed = await asyncio.to_thread(pack, folder)
        except Exception as error:
            print(f"[partial] {story_key[:8]}: pack failed, kept local: {error}", flush=True)
            return False
        if packed is None:
            return False
        flac, samples = packed
        key = storage.partial_key_for(story_key)
        try:
            await asyncio.to_thread(storage.upload_file, flac, key)
            # Saved before the folder goes, so a play from now on restores it.
            await story_audio_store.set_partial(story_key, key, samples)
        except Exception as error:
            print(f"[partial] {story_key[:8]}: upload failed, kept local: {error}", flush=True)
            return False
        finally:
            flac.unlink(missing_ok=True)
        if busy(story_key):
            # Played again meanwhile: the local chunks stay, the bucket copy is not needed.
            await story_audio_store.clear_partial(story_key)
            await delete_partial(story_key, key)
            return False
        await asyncio.to_thread(shutil.rmtree, folder, True)
        print(f"[partial] {story_key[:8]}: packed {len(samples)} chunks into {key}, local files removed", flush=True)
        return True


def may_be_packed(folder: Path) -> bool:
    # Chunk 0 on disk: the folder was never packed (pack removes all of it).
    return storage.storage_configured() and not _chunk_path(folder, 0).exists()


async def restore_partial(story_key: str, folder: Path) -> int:
    """Called before a story's chunks are generated: splits its packed partial back into `folder`.
    Returns how many chunks were restored. A missing or broken partial restores nothing, and the
    story is generated from its first missing chunk."""
    if not may_be_packed(folder):
        return 0
    async with _story_lock(story_key):
        partial = await story_audio_store.get_partial(story_key)
        if partial is None:
            return 0
        key, samples = partial
        flac = folder / PARTIAL_NAME
        restored = 0
        try:
            await asyncio.to_thread(storage.download_file, key, flac)
            restored = await asyncio.to_thread(unpack, flac, samples, folder)
            print(f"[partial] {story_key[:8]}: restored {restored} chunks from {key}", flush=True)
        except Exception as error:
            print(f"[partial] {story_key[:8]}: could not restore {key}, generating again: {error}", flush=True)
        finally:
            flac.unlink(missing_ok=True)
        # The chunks are local again (or lost): the bucket copy is no longer needed.
        await story_audio_store.clear_partial(story_key)
        await delete_partial(story_key, key)
        return restored
