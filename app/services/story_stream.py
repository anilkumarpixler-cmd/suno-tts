from __future__ import annotations


import asyncio
import hashlib
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from app.services.audio import AUDIO_PIPELINE_VERSION, CHUNK_PAUSE_MS, clean_chunk_wav, concat_wav_bytes
from app.services.chatterbox_client import generate_on_colab
from app.services.story_chunker import chunk_story
from app.store import TTS_DIR, ensure_dirs

# Keep strong references so background workers are not garbage-collected mid-run,
# and so a second request for the same story reuses the running worker.
_workers: dict[str, asyncio.Task[None]] = {}
_first_chunk_locks: dict[str, asyncio.Lock] = {}

# How often a held long-poll request re-checks the disk for its file.
STREAM_WAIT_POLL_S = 0.25

# Kaggle has 2 GPUs, so two chunks can be generated at once.
PARALLEL_CHUNKS = 2


@dataclass
class StoryChunkResult:
    index: int
    text: str
    path: Path
    is_final: bool = False


def normalize_story_value(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def story_cache_key(story_text: str, voice_id: str | None, language: str | None, voice_version: str | None) -> str:
    payload = "|".join(
        [
            normalize_story_value(story_text),
            normalize_story_value(voice_id or ""),
            normalize_story_value(language or ""),
            normalize_story_value(voice_version or ""),
            AUDIO_PIPELINE_VERSION,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def story_stream_dir(story_key: str) -> Path:
    ensure_dirs()
    return TTS_DIR / "story_streams" / story_key


def story_stream_final_path(story_key: str) -> Path:
    return story_stream_dir(story_key) / "final.wav"


def story_stream_chunk_path(story_key: str, chunk_index: int) -> Path:
    return story_stream_dir(story_key) / f"chunk_{chunk_index}.wav"


def story_stream_chunk_paths(story_key: str, count: int) -> list[Path]:
    return [story_stream_chunk_path(story_key, index) for index in range(count)]


def story_stream_chunk_exists(story_key: str, chunk_index: int) -> bool:
    path = story_stream_chunk_path(story_key, chunk_index)
    return path.exists() and path.stat().st_size > 0


def story_stream_error_path(story_key: str) -> Path:
    return story_stream_dir(story_key) / "error.txt"


def story_stream_error(story_key: str) -> str | None:
    path = story_stream_error_path(story_key)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or "Story stream generation failed"


async def wait_for_story_file(story_key: str, path: Path, timeout: float) -> str:
    """Long-poll helper: 'ready' once the file exists, 'failed' if generation errored,
    'pending' if neither happened within `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        if path.exists() and path.stat().st_size > 0:
            return "ready"
        if story_stream_error(story_key):
            return "failed"
        if time.monotonic() >= deadline:
            return "pending"
        await asyncio.sleep(STREAM_WAIT_POLL_S)


def _write_atomic(path: Path, data: bytes) -> None:
    # Write to a temp file first so readers never see a half-written WAV.
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def _wav_seconds(data: bytes) -> float:
    try:
        with wave.open(BytesIO(data), "rb") as src:
            return src.getnframes() / src.getframerate()
    except (wave.Error, EOFError):
        # Float WAVs are not readable by `wave`: use the header's byte rate and data size.
        byte_rate = int.from_bytes(data[28:32], "little")
        data_at = data.find(b"data")
        if not byte_rate or data_at < 0:
            return 0.0
        return int.from_bytes(data[data_at + 4:data_at + 8], "little") / byte_rate


async def generate_story_chunk_audio(
    chunk_text: str, language: str, prompt_wav_path: str | Path, is_last: bool
) -> bytes:
    from app.services.tts import language_id

    lang_id = language_id(language)
    started = time.monotonic()
    audio = await generate_on_colab(chunk_text.strip(), lang_id, str(prompt_wav_path))
    # Generation time vs. audio length: a ratio near or above 1 means the next chunk can't be
    # ready before the current one finishes playing.
    took = time.monotonic() - started
    seconds = _wav_seconds(audio)
    ratio = took / seconds if seconds else 0
    print(
        f"[stream] {len(chunk_text.split())} words: generated in {took:.1f}s, "
        f"audio {seconds:.1f}s, ratio {ratio:.2f}",
        flush=True,
    )
    # Every chunk but the last ends with a fixed pause, so chunk changes sound like a full stop.
    return clean_chunk_wav(audio, 0 if is_last else CHUNK_PAUSE_MS)


def _merge_chunk_wavs(chunk_paths: list[Path], dest: Path) -> Path:
    if not chunk_paths:
        raise RuntimeError("No story chunks to merge")
    chunk_data = [path.read_bytes() for path in chunk_paths if path.exists() and path.stat().st_size > 0]
    if not chunk_data:
        raise RuntimeError("No valid chunk audio found to assemble final story")
    return concat_wav_bytes(chunk_data, dest)


async def start_story_stream(
    story_text: str,
    language: str,
    voice_id: str | None,
    voice_version: str | None,
    prompt_wav_path: str | Path,
) -> dict[str, Any]:
    story_key = story_cache_key(story_text, voice_id, language, voice_version)
    final_path = story_stream_final_path(story_key)
    if final_path.exists() and final_path.stat().st_size > 0:
        return {
            "cached": True,
            "key": story_key,
            "final_path": str(final_path),
            "chunks": [],
        }

    chunks = chunk_story(story_text)
    if not chunks:
        return {
            "cached": False,
            "key": story_key,
            "final_path": str(final_path),
            "chunks": [],
        }

    stream_dir = story_stream_dir(story_key)
    stream_dir.mkdir(parents=True, exist_ok=True)

    first_path = story_stream_chunk_path(story_key, 0)
    # At most PARALLEL_CHUNKS chunks generate at once. Waiters get a slot first-come first-served
    # and tasks are created in index order, so chunks always start lowest index first. Each is
    # saved as chunk_{index}.wav and the app plays them by index, so playback stays in order.
    slots = asyncio.Semaphore(PARALLEL_CHUNKS)

    async def _generate_chunk(index: int) -> None:
        async with slots:
            if story_stream_chunk_exists(story_key, index):
                return
            audio = await generate_story_chunk_audio(chunks[index], language, prompt_wav_path, index == len(chunks) - 1)
            _write_atomic(story_stream_chunk_path(story_key, index), audio)

    async def _background_worker() -> None:
        try:
            # A failed chunk cancels the rest of the group.
            async with asyncio.TaskGroup() as group:
                for index in range(1, len(chunks)):
                    group.create_task(_generate_chunk(index))
            _merge_chunk_wavs(story_stream_chunk_paths(story_key, len(chunks)), final_path)
        except Exception as error:
            if isinstance(error, ExceptionGroup):
                error = error.exceptions[0]
            # Record the failure so clients stop polling instead of waiting forever.
            story_stream_error_path(story_key).write_text(str(error) or type(error).__name__, encoding="utf-8")
        finally:
            _workers.pop(story_key, None)

    # Start the worker before chunk 0, so chunk 1 is generated on the other GPU at the same time.
    running = _workers.get(story_key)
    if running is None or running.done():
        # A fresh request retries a previously failed stream.
        story_stream_error_path(story_key).unlink(missing_ok=True)
        _workers[story_key] = asyncio.create_task(_background_worker())

    lock = _first_chunk_locks.setdefault(story_key, asyncio.Lock())
    async with lock:
        # Takes the first slot: the worker task has not run yet, so chunk 1 gets the second.
        await _generate_chunk(0)

    return {
        "cached": False,
        "key": story_key,
        "first_chunk": {"index": 0, "path": str(first_path)},
        "final_path": str(final_path),
        "chunk_count": len(chunks),
        "chunk_words": [len(chunk.split()) for chunk in chunks],
    }


def cancel_story_stream(story_key: str) -> bool:
    """Stop generating a story the listener skipped. Chunks already made stay on disk, so
    playing it again later resumes from the first missing chunk."""
    running = _workers.get(story_key)
    if running is None or running.done():
        return False
    # CancelledError is not an Exception, so the worker writes no error.txt for this.
    running.cancel()
    return True


async def ensure_story_final_audio(
    story_text: str,
    language: str,
    voice_id: str | None,
    voice_version: str | None,
    prompt_wav_path: str | Path,
) -> dict[str, Any]:
    story_key = story_cache_key(story_text, voice_id, language, voice_version)
    final_path = story_stream_final_path(story_key)
    if final_path.exists() and final_path.stat().st_size > 0:
        return {
            "cached": True,
            "key": story_key,
            "path": str(final_path),
        }

    chunks = chunk_story(story_text)
    if not chunks:
        return {
            "cached": False,
            "key": story_key,
            "path": str(final_path),
        }

    stream_dir = story_stream_dir(story_key)
    stream_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for index, chunk in enumerate(chunks):
        chunk_path = story_stream_chunk_path(story_key, index)
        if not chunk_path.exists() or chunk_path.stat().st_size == 0:
            audio = await generate_story_chunk_audio(chunk, language, prompt_wav_path, index == len(chunks) - 1)
            chunk_path.write_bytes(audio)
        saved_paths.append(chunk_path)

    final_audio = _merge_chunk_wavs(saved_paths, final_path)
    return {
        "cached": False,
        "key": story_key,
        "path": str(final_audio),
    }
