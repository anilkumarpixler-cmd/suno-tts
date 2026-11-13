from __future__ import annotations


import asyncio
import bisect
import hashlib
import itertools
import time
import wave
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from app.services import hls, storage, story_audio_store, story_partial


from app.services.audio import (
    AUDIO_PIPELINE_VERSION,
    CHUNK_PAUSE_MS,
    clean_chunk_wav,
    compress_story_wav,
    concat_wav_bytes,
)
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

# While the listener plays, generate at most this many chunks past the one they are hearing, so a
# story skipped halfway wastes at most this many chunks of GPU time. 2, not more: an HLS player
# already fetches 2–3 segments ahead of what it plays, and each fetch moves this limit.
PLAY_AHEAD_CHUNKS = 2
# While the listener is paused, generate at most this many chunks past the one they paused on.
PAUSE_AHEAD_CHUNKS = 1
# story_key -> highest chunk index allowed to generate (absent = no stream running, no limit).
_ahead_limits: dict[str, int] = {}
# story_key -> chunk the listener is on, and stories whose listener is paused.
_listen_index: dict[str, int] = {}
_paused: set[str] = set()
# Replaced every time a limit moves, so a waiter is woken once and then re-checks its limit.
_limit_events: dict[str, asyncio.Event] = {}

# A stream waiting on its listener with no request for this story in this long (the app was closed
# mid-story) stops: the row becomes partial and the waiting worker no longer holds memory.
LISTENER_IDLE_S = 300 #300 seconds = 5 minutes
# Streams stopped that way, kept so a listener back from a long pause continues without the app
# sending the story again. Oldest dropped first.
STOPPED_STREAMS_KEPT = 1000

# play_id -> story_key for every play request still listening. One play is one tap on a story in
# the app, so a story is only stopped once nobody is listening to it any more: a late cancel from
# an earlier play (A -> B -> A) or from another device can't stop it.
_plays: dict[str, str] = {}
# Plays cancelled before their start request arrived: the app's cancel can overtake its own start.
_cancelled_plays: OrderedDict[str, None] = OrderedDict()
CANCELLED_PLAYS_KEPT = 1024
# play_id -> device that started it.
_play_devices: dict[str, str] = {}

# One phone plays one story at a time: device_id -> story_key it is generating. Starting another
# story stops this one unless another device still listens to it, so one phone can't fill the
# GPU queue for everyone (also when the app's cancel never arrives).
_device_stories: dict[str, str] = {}
# Removals of a story's local files once its device moved on, kept so they are not garbage-collected.
_cleanup_tasks: set[asyncio.Task[Any]] = set()
# Stories one phone may have prefetched at once (its next and previous neighbours); the oldest is
# dropped first. device_id -> story keys, oldest first.
PREFETCH_STORIES_PER_DEVICE = 2
_device_prefetches: dict[str, list[str]] = {}


def _wake_limit_waiters(story_key: str) -> None:
    event = _limit_events.pop(story_key, None)
    if event is not None:
        event.set()


def _set_ahead_limit(story_key: str) -> None:
    ahead = PAUSE_AHEAD_CHUNKS if story_key in _paused else PLAY_AHEAD_CHUNKS
    _ahead_limits[story_key] = _listen_index.get(story_key, 0) + ahead
    _wake_limit_waiters(story_key)


def _clear_ahead_limit(story_key: str) -> None:
    _ahead_limits.pop(story_key, None)
    _listen_index.pop(story_key, None)
    _paused.discard(story_key)
    _wake_limit_waiters(story_key)


async def _wait_for_listener(story_key: str, index: int) -> bool:
    """Chunks too far past the listener wait here, outside the GPU queue, until they catch up.
    False once nobody asked for this story for LISTENER_IDLE_S."""
    while (limit := _ahead_limits.get(story_key)) is not None and index > limit:
        try:
            # Every request for the story wakes this, so the timeout restarts.
            await asyncio.wait_for(_limit_events.setdefault(story_key, asyncio.Event()).wait(), LISTENER_IDLE_S)
        except TimeoutError:
            return False
    return True


def _stream_running(story_key: str) -> bool:
    running = _workers.get(story_key)
    return running is not None and not running.done()


async def keep_stream_alive(story_key: str) -> None:
    """The app asked about this story, so it is still open: restart the idle timer, or restart a
    stream it stopped."""
    if _stream_running(story_key):
        _wake_limit_waiters(story_key)
    elif story_key in _stopped_streams:
        await _restart_stopped_stream(story_key)


def note_listening(story_key: str, chunk_index: int) -> None:
    """The app fetched chunk `chunk_index` to play it: generate up to PLAY_AHEAD_CHUNKS past it."""
    if story_key not in _ahead_limits or not _stream_running(story_key):
        return
    _listen_index[story_key] = max(0, chunk_index)
    _set_ahead_limit(story_key)


def pause_story_stream(story_key: str, current_index: int) -> bool:
    """Listener paused on chunk `current_index`: stop generating PAUSE_AHEAD_CHUNKS past it.
    Chunks already running on the GPU still finish."""
    if not _stream_running(story_key):
        return False
    _paused.add(story_key)
    _listen_index[story_key] = max(0, current_index)
    _set_ahead_limit(story_key)
    return True


def resume_story_stream(story_key: str) -> bool:
    """Listener resumed: generate up to PLAY_AHEAD_CHUNKS past the chunk they are on again."""
    was_paused = story_key in _paused
    _paused.discard(story_key)
    if story_key in _ahead_limits:
        _set_ahead_limit(story_key)
    return was_paused and _stream_running(story_key)


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


async def story_ready(story_key: str, check_bucket: bool = False) -> bool:
    """The whole story is made: its final.wav is on disk, or (once story_cleanup removed the local
    files) the database says it is ready in the bucket. Then nothing is generated again.
    check_bucket: also ask the bucket, for a play about to use it (the long polls skip that call)."""
    final_path = story_stream_final_path(story_key)
    if final_path.exists() and final_path.stat().st_size > 0:
        return True
    if not storage.storage_configured():
        return False
    key = await story_audio_store.get_ready_key(story_key)
    if not storage.is_stored(key):
        return False
    if check_bucket and not await bucket_has_story(story_key, key):
        return False
    return True


async def bucket_has_story(story_key: str, key: str) -> bool:
    """False once the bucket lost the file (deleted by hand, say): the row is no longer ready, so
    the story is generated again. A bucket error counts as there, so an outage regenerates nothing."""
    try:
        if await asyncio.to_thread(storage.exists, key):
            return True
    except Exception as error:
        print(f"[storage] {story_key[:8]}: bucket check failed, trusting the database: {error}", flush=True)
        return True
    final_path = story_stream_final_path(story_key)
    if final_path.exists() and final_path.stat().st_size > 0:
        # Not cleaned up yet: the local copy still plays, the row stays as it is.
        print(f"[storage] {story_key[:8]}: {key} missing in the bucket, playing the local copy", flush=True)
        return False
    print(f"[storage] {story_key[:8]}: {key} missing in the bucket, generating it again", flush=True)
    await story_audio_store.mark_missing(story_key)
    return False


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
    # Devices that prefetched this chunk: a prefetch is dropped only once none of them needs it.
    prefetched_for: set[str | None] = field(default_factory=set)


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


# Strong refs to background segment encodes, so asyncio does not drop them mid-run.
_pre_encode_tasks: set[asyncio.Task] = set()


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
                # Encode the HLS segment now, not on the player's first request.
                _pre_encode_tasks.add(task := asyncio.create_task(hls.encode_ready(path.parent)))
                task.add_done_callback(_pre_encode_tasks.discard)
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
    device_id: str | None = None,
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
    if priority == PRIO_PREFETCH:
        job.prefetched_for.add(device_id)
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


def _release_prefetches(device_id: str | None, drop: Callable[[str], bool]) -> None:
    # Only this device's prefetches: another phone's next story keeps its chunks. A chunk several
    # devices prefetched stays queued until the last of them lets it go. Chunks on a GPU still finish.
    listened = set(_plays.values())
    for key, job in list(_jobs.items()):
        if (
            drop(job.story_key)
            and job.story_key not in listened
            and job.priority == PRIO_PREFETCH
            and not job.started
            and device_id in job.prefetched_for
        ):
            job.prefetched_for.discard(device_id)
            if not job.prefetched_for:
                job.cancelled = True
                job.future.cancel()
                _jobs.pop(key, None)


def _drop_stale_prefetches(keep_story_key: str, device_id: str | None = None) -> None:
    # A new story is starting, so neighbours prefetched for the previous one are no longer next:
    # drop them before they take a GPU ahead of this story's chunks.
    if device_id:
        _device_prefetches.pop(device_id, None)
    _release_prefetches(device_id, lambda story_key: story_key != keep_story_key)


def _leave_previous_story(device_id: str | None, story_key: str) -> None:
    # This device moved on to `story_key`: stop the story it played before, unless another device
    # still listens to it. Its own plays of that story end here, even if their cancel never comes.
    if not device_id:
        return
    previous = _device_stories.pop(device_id, None)
    if previous is None or previous == story_key:
        return
    for play, key in list(_plays.items()):
        if key == previous and _play_devices.get(play) == device_id:
            _plays.pop(play, None)
            _play_devices.pop(play, None)
    if previous in _plays.values() or previous in _device_stories.values():
        return
    if cancel_story_stream(previous, include_first=True):
        print(f"[stream] {previous[:8]}: device started another story, stopped", flush=True)
    # Imported here: story_cleanup imports this module.
    from app.services.story_cleanup import remove_left_story

    task = asyncio.create_task(remove_left_story(previous))
    _cleanup_tasks.add(task)
    task.add_done_callback(_cleanup_tasks.discard)


@dataclass
class _StreamArgs:
    chunks: list[str]
    language: str
    prompt_wav_path: str | Path
    story_id: int | None
    voice_id: str | None
    device_id: str | None


# story_key -> (what the stream needs, chunk the listener was on) for streams stopped by the idle timer.
_stopped_streams: OrderedDict[str, tuple[_StreamArgs, int]] = OrderedDict()


def _start_worker(story_key: str, args: _StreamArgs) -> None:
    # A fresh worker retries a previously failed stream.
    story_stream_error_path(story_key).unlink(missing_ok=True)
    _workers[story_key] = asyncio.create_task(_run_stream(story_key, args))


def _stop_idle_stream(story_key: str, args: _StreamArgs, worker: asyncio.Task[None]) -> None:
    if worker.done() or worker.cancelling():
        return
    _stopped_streams[story_key] = (args, _listen_index.get(story_key, 0))
    _stopped_streams.move_to_end(story_key)
    while len(_stopped_streams) > STOPPED_STREAMS_KEPT:
        _stopped_streams.popitem(last=False)
    print(f"[stream] {story_key[:8]}: nobody listening for {LISTENER_IDLE_S}s, stopped", flush=True)
    # Ends like a skip: the row becomes partial and the made chunks stay on disk.
    worker.cancel()


async def _restart_stopped_stream(story_key: str) -> None:
    entry = _stopped_streams.pop(story_key, None)
    if entry is None or _stream_running(story_key):
        return
    args, listen_index = entry
    _paused.discard(story_key)
    _listen_index[story_key] = listen_index
    _set_ahead_limit(story_key)
    # Started before anything is awaited, so a request meanwhile does not see the story as stopped.
    _start_worker(story_key, args)
    print(f"[stream] {story_key[:8]}: listener back, continuing from chunk {listen_index}", flush=True)
    if args.device_id and args.voice_id:
        await story_audio_store.start_audio(
            story_key, args.story_id, args.voice_id, args.language, args.device_id, len(args.chunks)
        )


def story_busy(story_key: str) -> bool:
    """The story is generating or a play still waits on it: its files must stay."""
    return (
        _stream_running(story_key)
        or story_key in _plays.values()
        or any(key == story_key for key, _ in _jobs)
    )


def story_on_device(story_key: str) -> bool:
    """A phone is still on this story (playing or paused) and has not started another one."""
    return story_key in _device_stories.values()


def story_in_use(story_key: str) -> bool:
    return story_busy(story_key) or story_on_device(story_key)


def story_stream_stopped(story_key: str, chunk_index: int) -> bool:
    """Nothing will make this chunk: no stream running or queued for it and none kept to restart
    (the server restarted, or the stream was stopped long ago). The app plays the story again."""
    return (
        not story_stream_chunk_exists(story_key, chunk_index)
        and not _stream_running(story_key)
        and (story_key, chunk_index) not in _jobs
        and story_key not in _stopped_streams
        and story_stream_error(story_key) is None
    )


async def _upload_final(story_key: str, final_path: Path) -> str:
    """The final_key to save: the bucket key of the compressed story once uploaded, else the local
    WAV. The local files stay until story_cleanup removes them: chunks may still be playing."""
    local_key = str(final_path.relative_to(TTS_DIR))
    if not storage.storage_configured():
        return local_key
    stored = await asyncio.to_thread(compress_story_wav, final_path)
    key = storage.story_key_for(story_key, stored)
    try:
        await asyncio.to_thread(storage.upload_file, stored, key)
    except Exception as error:
        # A bucket problem never fails the story: it plays from the local file.
        print(f"[storage] {story_key[:8]}: upload failed, kept local: {error}", flush=True)
        if stored != final_path:
            stored.unlink(missing_ok=True)
        return local_key
    print(
        f"[storage] {story_key[:8]}: uploaded {key} "
        f"({stored.stat().st_size // 1024} KB, WAV was {final_path.stat().st_size // 1024} KB)",
        flush=True,
    )
    return key


async def _restore_partial(story_key: str) -> None:
    """Chunks packed into the bucket when the story was left come back before any is generated.
    shield: a cancel meanwhile must not leave half a restore behind. Skipped without a yield when
    nothing can be packed, so callers keep their order in the GPU queue."""
    folder = story_stream_dir(story_key)
    if story_partial.may_be_packed(folder):
        await asyncio.shield(story_partial.restore_partial(story_key, folder))


async def _finish_story(story_key: str, final_path: Path, chunk_count: int) -> None:
    duration_ms = int(_wav_seconds(final_path.read_bytes()) * 1000)
    # Read before mark_ready clears it: a packed partial left over is no longer needed.
    partial = await story_audio_store.get_partial(story_key) if storage.storage_configured() else None
    await story_audio_store.mark_ready(
        story_key, await _upload_final(story_key, final_path), duration_ms, chunk_count
    )
    if partial:
        await story_partial.delete_partial(story_key, partial[0])


async def _run_stream(story_key: str, args: _StreamArgs) -> None:
    chunks = args.chunks
    final_path = story_stream_final_path(story_key)
    worker = asyncio.current_task()
    # At most GPU_COUNT of this story's chunks wait in the GPU queue at once. Waiters get a slot
    # first-come first-served and tasks are created in index order, so chunks always start lowest
    # index first. Each is saved as chunk_{index}.wav and the app plays them by index, so playback
    # stays in order. The prefetch GPU still takes the next story's prefetched chunks before these.
    feed = asyncio.Semaphore(GPU_COUNT)

    async def _generate_chunk(index: int) -> None:
        async with feed:
            if story_stream_chunk_exists(story_key, index):
                return
            # Held inside the slot so chunks still start lowest index first. A chunk too far past
            # the listener waits here, outside the GPU queue, so the GPUs stay free for other work.
            if not await _wait_for_listener(story_key, index):
                _stop_idle_stream(story_key, args, worker)
                raise asyncio.CancelledError
            # shield: cancelling this story must not cancel a job another caller is waiting on.
            await asyncio.shield(
                _submit_chunk(story_key, index, chunks, args.language, args.prompt_wav_path, PRIO_ACTIVE)
            )

    merged = False
    try:
        await _restore_partial(story_key)
        # A failed chunk cancels the rest of the group.
        async with asyncio.TaskGroup() as group:
            for index in range(1, len(chunks)):
                group.create_task(_generate_chunk(index))
        await asyncio.to_thread(_merge_chunk_wavs, story_stream_chunk_paths(story_key, len(chunks)), final_path)
        merged = True
        # The app can play the local file already, while it is compressed and uploaded.
        _notify_files_changed()
        # shield: a listener leaving during the upload must not stop it or mark the story partial.
        await asyncio.shield(_finish_story(story_key, final_path, len(chunks)))
    except asyncio.CancelledError:
        # The listener left: record which chunks are on disk so playing again resumes. Once merged,
        # the shielded finish marks the story ready instead.
        if not merged:
            saved = [path.name for path in story_stream_chunk_paths(story_key, len(chunks)) if path.exists()]
            await asyncio.shield(story_audio_store.mark_partial(story_key, saved))
        raise
    except Exception as error:
        if isinstance(error, ExceptionGroup):
            error = error.exceptions[0]
        # Record the failure so clients stop polling instead of waiting forever.
        story_stream_error_path(story_key).write_text(str(error) or type(error).__name__, encoding="utf-8")
        await story_audio_store.mark_failed(story_key)
        _notify_files_changed()
    finally:
        if _workers.get(story_key) is worker:
            _workers.pop(story_key, None)
            _clear_ahead_limit(story_key)
            # Nothing is left to generate, so there is nothing for these plays to stop. The device
            # stays on the story until it starts another one: its files are kept while it listens.
            for play, key in list(_plays.items()):
                if key == story_key:
                    _plays.pop(play, None)
                    _play_devices.pop(play, None)


async def start_story_stream(
    story_text: str,
    language: str,
    voice_id: str | None,
    voice_version: str | None,
    prompt_wav_path: str | Path,
    play_id: str | None = None,
    story_id: int | None = None,
    device_id: str | None = None,
    resume_ms: int = 0,
) -> dict[str, Any]:
    story_key = story_cache_key(story_text, voice_id, language, voice_version)
    if play_id and play_id in _cancelled_plays:
        # The listener already left this story before the request got here.
        return {"cancelled": True, "key": story_key, "chunks": []}

    _leave_previous_story(device_id, story_key)
    _drop_stale_prefetches(story_key, device_id)

    final_path = story_stream_final_path(story_key)
    if await story_ready(story_key, check_bucket=True):
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

    # One row per story audio; the audio itself stays on disk.
    if device_id and voice_id:
        await story_audio_store.start_audio(story_key, story_id, voice_id, language, device_id, len(chunks))

    stream_dir = story_stream_dir(story_key)
    stream_dir.mkdir(parents=True, exist_ok=True)

    first_path = story_stream_chunk_path(story_key, 0)

    # Registered before anything is awaited, so a cancel from an earlier play can't stop this one.
    if play_id:
        _plays[play_id] = story_key
        if device_id:
            _play_devices[play_id] = device_id
    if device_id:
        _device_stories[device_id] = story_key

    # A new play starts at chunk 0, and a pause left by a closed app must not block it.
    _stopped_streams.pop(story_key, None)
    _paused.discard(story_key)
    _listen_index[story_key] = 0
    _set_ahead_limit(story_key)

    # Start the worker before chunk 0, so chunk 1 is generated on the other GPU at the same time.
    running = _workers.get(story_key)
    # A worker that was just cancelled is still finishing: it must not be reused.
    if running is None or running.done() or running.cancelling():
        _start_worker(story_key, _StreamArgs(chunks, language, prompt_wav_path, story_id, voice_id, device_id))

    # A story left halfway and packed into the bucket: its chunks come back first, so chunk 0 is not
    # generated again. The worker waits on the same restore.
    await _restore_partial(story_key)
    if play_id and play_id not in _plays and not story_stream_chunk_exists(story_key, 0):
        # The listener left while it was restoring.
        return {"cancelled": True, "key": story_key, "chunks": []}

    # Resuming halfway: generation continues from the listener's chunk, not from chunk 0.
    start_chunk, start_ms, resume_reset = resume_point(story_key, resume_ms)
    start_chunk = min(start_chunk, len(chunks) - 1)
    if start_chunk:
        _listen_index[story_key] = start_chunk
        _set_ahead_limit(story_key)

    # Instant if the chunk was prefetched (or already made); if the prefetch is still queued or
    # generating, this waits on that same job (moved ahead of everything else) instead of making it again.
    first = _submit_chunk(story_key, start_chunk, chunks, language, prompt_wav_path, PRIO_FIRST)
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
        "start_chunk": start_chunk,
        "start_ms": start_ms,
        "resume_reset": resume_reset,
    }


def resume_point(story_key: str, resume_ms: int) -> tuple[int, int, bool]:
    """(chunk, its start in ms, reset) to continue a story at `resume_ms`. The listener only ever
    heard made chunks, so the position is normally inside them. If not (a restore failed), the story
    continues at the first chunk not made yet and `reset` tells the app its position moved."""
    if resume_ms <= 0:
        return 0, 0, False
    folder = story_stream_dir(story_key)
    found = hls.chunk_at(folder, resume_ms)
    if found is not None:
        return found[0], found[1], False
    return hls.made_count(folder), hls.made_ms(folder), True


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
    device_id = _play_devices.pop(play_id, None)
    if device_id and story_key is not None and _device_stories.get(device_id) == story_key:
        _device_stories.pop(device_id, None)
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
    await _restore_partial(story_key)
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


async def prefetch_story_start(
    story_text: str,
    language: str,
    voice_id: str | None,
    voice_version: str | None,
    prompt_wav_path: str | Path,
    device_id: str | None = None,
) -> str:
    """Queue the first PREFETCH_CHUNKS chunks of the story that will play next, so it starts
    instantly and keeps playing after a switch. The rest is generated once it is actually played."""
    story_key = story_cache_key(story_text, voice_id, language, voice_version)
    if await story_ready(story_key, check_bucket=True):
        return story_key
    chunks = chunk_story(story_text)
    if chunks:
        story_stream_dir(story_key).mkdir(parents=True, exist_ok=True)
        await _restore_partial(story_key)
        # Same priority runs in submit order, so chunk 0 is made before chunk 1.
        for index in range(min(PREFETCH_CHUNKS, len(chunks))):
            _submit_chunk(story_key, index, chunks, language, prompt_wav_path, PRIO_PREFETCH, device_id)
        if device_id:
            prefetched = _device_prefetches.setdefault(device_id, [])
            if story_key in prefetched:
                prefetched.remove(story_key)
            prefetched.append(story_key)
            while len(prefetched) > PREFETCH_STORIES_PER_DEVICE:
                oldest = prefetched.pop(0)
                _release_prefetches(device_id, lambda key: key == oldest)
    return story_key
