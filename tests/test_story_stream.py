import asyncio

from app.services import story_stream
from app.services.story_stream import story_cache_key, wait_for_story_file


def test_story_cache_key_is_stable() -> None:
    first = story_cache_key("A story for testing.", "voice-1", "English", "voice-v2")
    second = story_cache_key("A story for testing.", "voice-1", "English", "voice-v2")
    assert first == second
    assert first


def test_story_cache_key_changes_for_voice_or_story() -> None:
    base = story_cache_key("A story for testing.", "voice-1", "English", "voice-v2")
    voice_change = story_cache_key("A story for testing.", "voice-2", "English", "voice-v2")
    story_change = story_cache_key("Different story text.", "voice-1", "English", "voice-v2")
    assert base != voice_change
    assert base != story_change


def _stream_dir(tmp_path, monkeypatch, key: str):
    monkeypatch.setattr(story_stream, "TTS_DIR", tmp_path)
    directory = story_stream.story_stream_dir(key)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def test_wait_for_story_file_ready(tmp_path, monkeypatch) -> None:
    chunk = _stream_dir(tmp_path, monkeypatch, "k1") / "chunk_0.wav"
    chunk.write_bytes(b"RIFF")
    assert asyncio.run(wait_for_story_file("k1", chunk, 1.0)) == "ready"


def test_wait_for_story_file_failed(tmp_path, monkeypatch) -> None:
    directory = _stream_dir(tmp_path, monkeypatch, "k2")
    (directory / "error.txt").write_text("colab down", encoding="utf-8")
    assert asyncio.run(wait_for_story_file("k2", directory / "chunk_0.wav", 1.0)) == "failed"


def test_wait_for_story_file_pending_after_timeout(tmp_path, monkeypatch) -> None:
    directory = _stream_dir(tmp_path, monkeypatch, "k3")
    assert asyncio.run(wait_for_story_file("k3", directory / "chunk_0.wav", 0.3)) == "pending"


def test_wait_for_story_file_sees_file_written_while_waiting(tmp_path, monkeypatch) -> None:
    chunk = _stream_dir(tmp_path, monkeypatch, "k4") / "chunk_1.wav"

    async def scenario() -> str:
        async def write_later() -> None:
            await asyncio.sleep(0.3)
            chunk.write_bytes(b"RIFF")

        writer = asyncio.create_task(write_later())
        status = await wait_for_story_file("k4", chunk, 5.0)
        await writer
        return status

    assert asyncio.run(scenario()) == "ready"


def test_cancel_story_stream_stops_worker_without_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(story_stream, "TTS_DIR", tmp_path)

    async def fake_chunk(chunk_text, language, prompt_wav_path, is_last) -> bytes:
        if chunk_text.startswith("First"):
            return b"RIFF"
        await asyncio.sleep(30)  # stands in for a slow Colab chunk
        return b"RIFF"

    monkeypatch.setattr(story_stream, "generate_story_chunk_audio", fake_chunk)
    monkeypatch.setattr(story_stream, "chunk_story", lambda text: ["First chunk.", "Second chunk.", "Third chunk."])

    async def scenario() -> None:
        output = await story_stream.start_story_stream("story", "English", "v1", "default", "prompt.wav")
        key = output["key"]
        worker = story_stream._workers[key]
        await asyncio.sleep(0.05)
        assert story_stream.cancel_story_stream(key) is True
        await asyncio.gather(worker, return_exceptions=True)
        assert worker.cancelled()
        assert key not in story_stream._workers
        assert story_stream.story_stream_error(key) is None
        assert story_stream.story_stream_chunk_exists(key, 0)
        assert not story_stream.story_stream_chunk_exists(key, 1)
        # Nothing left to cancel.
        assert story_stream.cancel_story_stream(key) is False

    asyncio.run(scenario())


def _fresh_scheduler(tmp_path, monkeypatch) -> None:
    # Each asyncio.run has its own loop: start every test with an empty queue and no workers.
    monkeypatch.setattr(story_stream, "TTS_DIR", tmp_path)
    monkeypatch.setattr(story_stream, "_pending", [])
    monkeypatch.setattr(story_stream, "_work", None)
    monkeypatch.setattr(story_stream, "_gpu_tasks", [])
    monkeypatch.setattr(story_stream, "_jobs", {})
    monkeypatch.setattr(story_stream, "_workers", {})
    monkeypatch.setattr(story_stream, "_plays", {})
    monkeypatch.setattr(story_stream, "_cancelled_plays", story_stream.OrderedDict())

    async def fake_chunk(chunk_text, language, prompt_wav_path, is_last) -> bytes:
        if chunk_text.startswith("First"):
            return b"RIFF"
        await asyncio.sleep(30)  # stands in for a slow Colab chunk
        return b"RIFF"

    monkeypatch.setattr(story_stream, "generate_story_chunk_audio", fake_chunk)
    monkeypatch.setattr(story_stream, "chunk_story", lambda text: ["First chunk.", "Second chunk.", "Third chunk."])


def test_late_cancel_of_earlier_play_keeps_story_generating(tmp_path, monkeypatch) -> None:
    # A -> B -> A: the cancel of the first play of A must not stop the second one.
    _fresh_scheduler(tmp_path, monkeypatch)

    async def scenario() -> None:
        await story_stream.start_story_stream("story", "English", "v1", "default", "prompt.wav", play_id="a1")
        output = await story_stream.start_story_stream("story", "English", "v1", "default", "prompt.wav", play_id="a2")
        key = output["key"]
        worker = story_stream._workers[key]
        assert story_stream.cancel_story_play("a1") is False
        await asyncio.sleep(0.05)
        assert not worker.done()
        assert story_stream._workers[key] is worker
        # The last listener leaving stops it.
        assert story_stream.cancel_story_play("a2") is True
        await asyncio.gather(worker, return_exceptions=True)
        assert worker.cancelled()

    asyncio.run(scenario())


def test_cancel_while_preparing_drops_first_chunk(tmp_path, monkeypatch) -> None:
    # The listener skips a story before its first part is ready: the request answers "cancelled".
    _fresh_scheduler(tmp_path, monkeypatch)

    async def scenario() -> None:
        # Keep both GPUs busy with another story, so chunk 0 of this one is still queued.
        busy = ["Busy one.", "Busy two."]
        for index in range(2):
            story_stream._submit_chunk("other", index, busy, "English", "prompt.wav", story_stream.PRIO_FIRST)
        await asyncio.sleep(0.01)
        start = asyncio.create_task(
            story_stream.start_story_stream("story", "English", "v1", "default", "prompt.wav", play_id="p1")
        )
        await asyncio.sleep(0.05)
        key = story_stream.story_cache_key("story", "v1", "English", "default")
        assert story_stream._plays == {"p1": key}
        story_stream.cancel_story_play("p1")
        output = await asyncio.wait_for(start, 1.0)
        assert output["cancelled"] is True
        assert story_stream.story_stream_error(key) is None

    asyncio.run(scenario())


def test_starting_a_story_drops_stale_prefetches(tmp_path, monkeypatch) -> None:
    # Neighbours prefetched for the previous story must not take a GPU ahead of the new one.
    _fresh_scheduler(tmp_path, monkeypatch)

    async def scenario() -> None:
        # Keep both GPUs busy, so the prefetched chunks are still queued.
        busy = ["Busy one.", "Busy two."]
        for index in range(2):
            story_stream._submit_chunk("other", index, busy, "English", "prompt.wav", story_stream.PRIO_FIRST)
        await asyncio.sleep(0.01)
        stale = story_stream.prefetch_story_start("stale story", "English", "v1", "default", "prompt.wav")
        kept = story_stream.prefetch_story_start("next story", "English", "v1", "default", "prompt.wav")
        stale_futures = [story_stream._jobs[(stale, index)].future for index in range(story_stream.PREFETCH_CHUNKS)]
        start = asyncio.create_task(
            story_stream.start_story_stream("next story", "English", "v1", "default", "prompt.wav", play_id="p1")
        )
        await asyncio.sleep(0.05)
        assert all(future.cancelled() for future in stale_futures)
        assert not any(key[0] == stale for key in story_stream._jobs)
        # The story being started keeps its prefetched chunks.
        assert (kept, 0) in story_stream._jobs
        assert (kept, 1) in story_stream._jobs
        start.cancel()
        await asyncio.gather(start, return_exceptions=True)

    asyncio.run(scenario())


def test_cancel_that_overtakes_its_start_request(tmp_path, monkeypatch) -> None:
    _fresh_scheduler(tmp_path, monkeypatch)

    async def scenario() -> None:
        assert story_stream.cancel_story_play("early") is False
        output = await story_stream.start_story_stream("story", "English", "v1", "default", "prompt.wav", play_id="early")
        assert output["cancelled"] is True
        assert output["key"] not in story_stream._workers

    asyncio.run(scenario())


def test_replay_right_after_cancel_starts_a_new_worker(tmp_path, monkeypatch) -> None:
    # A worker that was just cancelled is still finishing: a new play must not reuse it.
    _fresh_scheduler(tmp_path, monkeypatch)

    async def scenario() -> None:
        output = await story_stream.start_story_stream("story", "English", "v1", "default", "prompt.wav", play_id="a1")
        key = output["key"]
        old = story_stream._workers[key]
        await asyncio.sleep(0.05)
        assert story_stream.cancel_story_play("a1") is True
        await story_stream.start_story_stream("story", "English", "v1", "default", "prompt.wav", play_id="a2")
        new = story_stream._workers[key]
        assert new is not old
        await asyncio.gather(old, return_exceptions=True)
        # The old worker finishing must not remove the new one.
        assert story_stream._workers.get(key) is new
        assert story_stream._plays == {"a2": key}
        new.cancel()
        await asyncio.gather(new, return_exceptions=True)

    asyncio.run(scenario())


def _running(story_key: str) -> list[int]:
    return sorted(index for (key, index), job in story_stream._jobs.items() if key == story_key and job.started)


def test_prefetch_takes_only_one_gpu(tmp_path, monkeypatch) -> None:
    # The next story is made on one GPU only; the other keeps making the playing story.
    _fresh_scheduler(tmp_path, monkeypatch)
    monkeypatch.setattr(story_stream, "chunk_story", lambda text: ["Slow one.", "Slow two.", "Slow three."])

    async def scenario() -> None:
        playing = ["Slow one.", "Slow two.", "Slow three."]
        for index in range(1, 3):
            story_stream._submit_chunk("playing", index, playing, "English", "prompt.wav", story_stream.PRIO_ACTIVE)
        nxt = story_stream.prefetch_story_start("next story", "English", "v1", "default", "prompt.wav")
        await asyncio.sleep(0.05)
        assert _running(nxt) == [0]
        assert _running("playing") == [1]
        # Both GPUs are busy, so chunk 2 of the playing story and chunk 1 of the next one wait.
        assert not story_stream._jobs[("playing", 2)].started
        assert not story_stream._jobs[(nxt, 1)].started

    asyncio.run(scenario())


def test_prefetch_gpu_helps_playing_story_when_idle(tmp_path, monkeypatch) -> None:
    # With nothing to prefetch, both GPUs make the playing story.
    _fresh_scheduler(tmp_path, monkeypatch)

    async def scenario() -> None:
        playing = ["Slow one.", "Slow two.", "Slow three."]
        for index in range(3):
            story_stream._submit_chunk("playing", index, playing, "English", "prompt.wav", story_stream.PRIO_ACTIVE)
        await asyncio.sleep(0.05)
        assert _running("playing") == [0, 1]

    asyncio.run(scenario())
