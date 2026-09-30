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
