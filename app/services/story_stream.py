from __future__ import annotations


import asyncio
import bisect
import hashlib
import itertools
import time
import wave
from collections import OrderedDict
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

# Held long-poll requests wake as soon as a chunk/final/error file is written; this is only the
# fallback re-check of the disk in case a file appears some other way.
STREAM_WAIT_POLL_S = 1.0
# Replaced on every write, so a waiter that grabbed the old one is woken exactly once.
# Kept with its event loop: an asyncio.Event only works on the loop it was first awaited on.
_files_changed: tuple[asyncio.AbstractEventLoop, asyncio.Event] | None = None


def _files_changed_event() -> asyncio.Event:
    global _files_changed
    loop = asyncio.get_running_loop()
    if _files_changed is None or _files_changed[0] is not loop:
        _files_changed = (loop, asyncio.Event())
    return _files_changed[1]


def _notify_files_changed() -> None:
    global _files_changed
    if _files_changed is not None:
        _files_changed[1].set()
        _files_changed = None

# Kaggle has 2 GPUs, so two chunks can be generated at once. The first GPU only makes the playing
# story; the other makes the next story's prefetch first and helps the playing story otherwise.
GPU_COUNT = 2

# Order in which a free GPU picks jobs (lowest first).
PRIO_FIRST = 0     # chunk 0 of a story the listener just started: they hear silence until it is ready
PRIO_PREFETCH = 1  # first chunks of the next story, made while the current one plays
PRIO_ACTIVE = 2    # remaining chunks of a playing story

# Chunks of the next story made ahead, one at a time on the prefetch GPU. The app asks for them
# once the playing story has 1 chunk buffered.
PREFETCH_CHUNKS = 2

# While the listener is paused, generate at most this many chunks past the one they paused on.
PAUSE_AHEAD_CHUNKS = 1  # after that, apply 2 instead
# story_key -> highest chunk index allowed while paused (absent = not paused).
_pause_limits: dict[str, int] = {}
_resume_events: dict[str, asyncio.Event] = {}

# play_id -> story_key for every play request still listening. One play is one tap on a story in
# the app, so a story is only stopped once nobody is listening to it any more: a late cancel from
# an earlier play (A -> B -> A) or from another device can't stop it.
_plays: dict[str, str] = {}
# Plays cancelled before their start request arrived: the app's cancel can overtake its own start.
_cancelled_plays: OrderedDict[str, None] = OrderedDict()
CANCELLED_PLAYS_KEPT = 1024


def _resume_event(story_key: str) -> asyncio.Event:
    return _resume_events.setdefault(story_key, asyncio.Event())


async def _wait_while_paused(story_key: str, index: int) -> None:
    while (limit := _pause_limits.get(story_key)) is not None and index > limit:
        await _resume_event(story_key).wait()


def pause_story_stream(story_key: str, current_index: int) -> bool:
    """Listener paused on chunk `current_index`: stop generating PAUSE_AHEAD_CHUNKS past it.
    Chunks already running on the GPU still finish."""
    running = _workers.get(story_key)
    if running is None or running.done():
        return False
    _pause_limits[story_key] = max(0, current_index) + PAUSE_AHEAD_CHUNKS
    _resume_event(story_key).clear()
    return True


def resume_story_stream(story_key: str) -> bool:
    """Listener resumed: generate the rest, starting from the first missing chunk."""
    was_paused = _pause_limits.pop(story_key, None) is not None
    event = _resume_events.get(story_key)
    if event is not None:
        event.set()
    running = _workers.get(story_key)
    return was_paused and running is not None and not running.done()


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
        # Taken before the checks, so a write between the check and the wait is not missed.
        changed = _files_changed_event()
        if path.exists() and path.stat().st_size > 0:
            return "ready"
        if story_stream_error(story_key):
            return "failed"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "pending"
        try:
            await asyncio.wait_for(changed.wait(), min(remaining, STREAM_WAIT_POLL_S))
        except TimeoutError:
            pass


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
    # In a thread: the sample loops would otherwise hold up every other request.
    return await asyncio.to_thread(clean_chunk_wav, audio, 0 if is_last else CHUNK_PAUSE_MS)


def _merge_chunk_wavs(chunk_paths: list[Path], dest: Path) -> Path:
    if not chunk_paths:
        raise RuntimeError("No story chunks to merge")
    chunk_data = [path.read_bytes() for path in chunk_paths if path.exists() and path.stat().st_size > 0]
    if not chunk_data:
        raise RuntimeError("No valid chunk audio found to assemble final story")
    return concat_wav_bytes(chunk_data, dest)


@dataclass
class _ChunkJob:
    story_key: str
    index: int
    text: str
    language: str
    prompt_wav_path: str
    is_last: bool
    priority: int
    future: asyncio.Future[Path]
    started: bool = False
    cancelled: bool = False


# One job per (story_key, chunk index): a chunk that is queued or generating is never queued twice,
# every caller that needs it waits on the same future.
_jobs: dict[tuple[str, int], _ChunkJob] = {}
# Waiting jobs as (priority, order, job), kept sorted; `_work` wakes the GPU loops when one is added.
_pending: list[tuple[int, int, _ChunkJob]] = []
_work: asyncio.Event | None = None
# Tie-breaker: jobs of the same priority run first come first served.
_job_order = itertools.count()
# Strong references, so the GPU loops are not garbage-collected.
_gpu_tasks: list[asyncio.Task[None]] = []


def _ensure_gpu_workers() -> asyncio.Event:
    global _work
    if _work is None:
        _work = asyncio.Event()
        for gpu in range(GPU_COUNT):
            # Only GPUs after the first make prefetches, so one GPU always makes the playing story.
            _gpu_tasks.append(asyncio.create_task(_gpu_loop(_work, allow_prefetch=gpu > 0)))
    return _work


def _push_job(priority: int, job: _ChunkJob) -> None:
    bisect.insort(_pending, (priority, next(_job_order), job), key=lambda entry: entry[:2])
    _ensure_gpu_workers().set()


def _take_job(allow_prefetch: bool) -> _ChunkJob | None:
    """Remove and return the most important waiting job this GPU may make, if any."""
    # Stale entries: the job was pushed again at a higher priority, started, or its story cancelled.
    _pending[:] = [
        (priority, order, job) for priority, order, job in _pending
        if not (job.started or job.cancelled or job.future.done() or priority != job.priority)
    ]
    for position, (_, _, job) in enumerate(_pending):
        if allow_prefetch or job.priority != PRIO_PREFETCH:
            del _pending[position]
            return job
    return None


async def _gpu_loop(work: asyncio.Event, allow_prefetch: bool) -> None:
    # One loop per GPU: take the most important waiting job it may make, generate it, repeat. When
    # the next story's prefetch is done, the prefetch GPU goes back to the playing story's chunks.
    while True:
        job = _take_job(allow_prefetch)
        if job is None:
            # Nothing between the check and the wait yields, so a job added meanwhile is not missed.
            work.clear()
            await work.wait()
            continue
        job.started = True
        try:
            path = story_stream_chunk_path(job.story_key, job.index)
            if not story_stream_chunk_exists(job.story_key, job.index):
                audio = await generate_story_chunk_audio(job.text, job.language, job.prompt_wav_path, job.is_last)
                await asyncio.to_thread(_write_atomic, path, audio)
                _notify_files_changed()
            job.future.set_result(path)
        except Exception as error:
            job.future.set_exception(error)
        finally:
            _jobs.pop((job.story_key, job.index), None)


def _submit_chunk(
    story_key: str,
    index: int,
    chunks: list[str],
    language: str,
    prompt_wav_path: str | Path,
    priority: int,
) -> asyncio.Future[Path]:
    """The only way a chunk gets generated. Returns the future of the job that makes it."""
    _ensure_gpu_workers()
    loop = asyncio.get_running_loop()
    if story_stream_chunk_exists(story_key, index):
        done: asyncio.Future[Path] = loop.create_future()
        done.set_result(story_stream_chunk_path(story_key, index))
        return done
    job = _jobs.get((story_key, index))
    if job is None:
        job = _ChunkJob(
            story_key, index, chunks[index], language, str(prompt_wav_path),
            index == len(chunks) - 1, priority, loop.create_future(),
        )
        # A prefetch is never awaited: read its error so asyncio does not log "never retrieved".
        job.future.add_done_callback(lambda future: future.cancelled() or future.exception())
        _jobs[(story_key, index)] = job
        _push_job(priority, job)
    elif priority < job.priority and not job.started:
        # A prefetched chunk became the story being played: queue it again ahead of the rest.
        job.priority = priority
        _push_job(priority, job)
    return job.future


def _drop_queued_jobs(story_key: str, include_first: bool = False) -> None:
    # Chunks already on a GPU keep running and are saved, so replaying the story is instant.
    # Chunk 0 is kept unless nobody is listening: a play request may be waiting on it.
    first = 0 if include_first else 1
    for key, job in list(_jobs.items()):
        if job.story_key == story_key and job.index >= first and not job.started:
            job.cancelled = True
            job.future.cancel()
            _jobs.pop(key, None)


def _drop_stale_prefetches(keep_story_key: str) -> None:
    # A new story is starting, so neighbours prefetched for the previous one are no longer next:
    # drop them before they take a GPU ahead of this story's chunks. Chunks already on a GPU still
    # finish. Global for now (one listener); with many users this must be scoped per app session.
    listened = set(_plays.values())
    for key, job in list(_jobs.items()):
        if (
            job.story_key != keep_story_key
            and job.story_key not in listened
            and job.priority == PRIO_PREFETCH
            and not job.started
        ):
            job.cancelled = True
            job.future.cancel()
            _jobs.pop(key, None)


async def start_story_stream(
    story_text: str,
    language: str,
    voice_id: str | None,
    voice_version: str | None,
    prompt_wav_path: str | Path,
    play_id: str | None = None,
) -> dict[str, Any]:
    story_key = story_cache_key(story_text, voice_id, language, voice_version)
    if play_id and play_id in _cancelled_plays:
        # The listener already left this story before the request got here.
        return {"cancelled": True, "key": story_key, "chunks": []}

    _drop_stale_prefetches(story_key)

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
    # At most GPU_COUNT of this story's chunks wait in the GPU queue at once. Waiters get a slot
    # first-come first-served and tasks are created in index order, so chunks always start lowest
    # index first. Each is saved as chunk_{index}.wav and the app plays them by index, so playback
    # stays in order. The prefetch GPU still takes the next story's prefetched chunks before these.
    feed = asyncio.Semaphore(GPU_COUNT)

    async def _generate_chunk(index: int) -> None:
        async with feed:
            if story_stream_chunk_exists(story_key, index):
                return
            # Held inside the slot so chunks still start lowest index first after a resume. A paused
            # story waits here, outside the GPU queue, so the GPUs stay free for other work.
            await _wait_while_paused(story_key, index)
            # shield: cancelling this story must not cancel a job another caller is waiting on.
            await asyncio.shield(_submit_chunk(story_key, index, chunks, language, prompt_wav_path, PRIO_ACTIVE))

    async def _background_worker() -> None:
        try:
            # A failed chunk cancels the rest of the group.
            async with asyncio.TaskGroup() as group:
                for index in range(1, len(chunks)):
                    group.create_task(_generate_chunk(index))
            await asyncio.to_thread(_merge_chunk_wavs, story_stream_chunk_paths(story_key, len(chunks)), final_path)
            _notify_files_changed()
        except Exception as error:
            if isinstance(error, ExceptionGroup):
                error = error.exceptions[0]
            # Record the failure so clients stop polling instead of waiting forever.
            story_stream_error_path(story_key).write_text(str(error) or type(error).__name__, encoding="utf-8")
            _notify_files_changed()
        finally:
            if _workers.get(story_key) is asyncio.current_task():
                _workers.pop(story_key, None)
                _pause_limits.pop(story_key, None)
                _resume_events.pop(story_key, None)
                # Nothing is left to generate, so there is nothing for these plays to stop.
                for play, key in list(_plays.items()):
                    if key == story_key:
                        _plays.pop(play, None)

    # Registered before anything is awaited, so a cancel from an earlier play can't stop this one.
    if play_id:
        _plays[play_id] = story_key

    # A new play request means someone is listening: a pause left by a closed app must not block it.
    resume_story_stream(story_key)

    # Start the worker before chunk 0, so chunk 1 is generated on the other GPU at the same time.
    running = _workers.get(story_key)
    # A worker that was just cancelled is still finishing: it must not be reused.
    if running is None or running.done() or running.cancelling():
        # A fresh request retries a previously failed stream.
        story_stream_error_path(story_key).unlink(missing_ok=True)
        _workers[story_key] = asyncio.create_task(_background_worker())

    # Instant if the chunk was prefetched; if the prefetch is still queued or generating, this waits
    # on that same job (moved ahead of everything else) instead of generating chunk 0 again.
    first = _submit_chunk(story_key, 0, chunks, language, prompt_wav_path, PRIO_FIRST)
    try:
        await asyncio.shield(first)
    except asyncio.CancelledError:
        # Chunk 0 was dropped because every listener left: answer instead of failing the request.
        if first.cancelled():
            return {"cancelled": True, "key": story_key, "chunks": []}
        raise

    return {
        "cached": False,
        "key": story_key,
        "first_chunk": {"index": 0, "path": str(first_path)},
        "final_path": str(final_path),
        "chunk_count": len(chunks),
        "chunk_words": [len(chunk.split()) for chunk in chunks],
    }


def cancel_story_stream(story_key: str, include_first: bool = False) -> bool:
    """Stop generating a story the listener skipped. Chunks already made stay on disk, so
    playing it again later resumes from the first missing chunk."""
    resume_story_stream(story_key)
    _drop_queued_jobs(story_key, include_first)
    running = _workers.get(story_key)
    if running is None or running.done():
        return False
    # CancelledError is not an Exception, so the worker writes no error.txt for this.
    running.cancel()
    return True


def cancel_story_play(play_id: str) -> bool:
    """The listener left one play (skip, new story, failure). The story stops generating only
    when no other play is listening to it. Works while the play is still waiting for chunk 0."""
    story_key = _plays.pop(play_id, None)
    if story_key is None:
        # Not started yet (the cancel overtook the start request) or already finished.
        _cancelled_plays[play_id] = None
        while len(_cancelled_plays) > CANCELLED_PLAYS_KEPT:
            _cancelled_plays.popitem(last=False)
        return False
    if story_key in _plays.values():
        return False
    return cancel_story_stream(story_key, include_first=True)


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

    # Goes through the GPU queue too, so it shares chunks with a stream of the same story.
    for index in range(len(chunks)):
        saved_paths.append(
            await asyncio.shield(_submit_chunk(story_key, index, chunks, language, prompt_wav_path, PRIO_ACTIVE))
        )

    final_audio = await asyncio.to_thread(_merge_chunk_wavs, saved_paths, final_path)
    _notify_files_changed()
    return {
        "cached": False,
        "key": story_key,
        "path": str(final_audio),
    }


def prefetch_story_start(
    story_text: str,
    language: str,
    voice_id: str | None,
    voice_version: str | None,
    prompt_wav_path: str | Path,
) -> str:
    """Queue the first PREFETCH_CHUNKS chunks of the story that will play next, so it starts
    instantly and keeps playing after a switch. The rest is generated once it is actually played."""
    story_key = story_cache_key(story_text, voice_id, language, voice_version)
    final_path = story_stream_final_path(story_key)
    if final_path.exists() and final_path.stat().st_size > 0:
        return story_key
    chunks = chunk_story(story_text)
    if chunks:
        story_stream_dir(story_key).mkdir(parents=True, exist_ok=True)
        # Same priority runs in submit order, so chunk 0 is made before chunk 1.
        for index in range(min(PREFETCH_CHUNKS, len(chunks))):
            _submit_chunk(story_key, index, chunks, language, prompt_wav_path, PRIO_PREFETCH)
    return story_key
