import asyncio
import shutil
import wave
from datetime import datetime, timedelta, timezone

import pytest

from app.services import audio, story_cleanup, story_partial, story_stream

needs_ffmpeg = pytest.mark.skipif(audio._ffmpeg_bin() is None, reason="ffmpeg not installed")


def _write_wav(path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(b"\x00\x01" * int(24000 * seconds))


def _write_noise_wav(path, frames: int, seed: int) -> None:
    # Varied samples, so a misplaced split between chunks would show.
    path.parent.mkdir(parents=True, exist_ok=True)
    data = bytes((seed * 31 + i * 7 + (i * i) % 251) % 256 for i in range(frames * 2))
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(data)


def _frames(path) -> bytes:
    with wave.open(str(path), "rb") as src:
        return src.readframes(src.getnframes())


async def _no_partial_rows(keys):
    return []


def test_compress_story_wav_makes_m4a(tmp_path) -> None:
    wav = tmp_path / "final.wav"
    _write_wav(wav)
    stored = audio.compress_story_wav(wav)
    assert stored.name == "final.m4a"
    assert 0 < stored.stat().st_size < wav.stat().st_size
    assert not list(tmp_path.glob("*.tmp*"))


def test_compress_story_wav_falls_back_to_flac(tmp_path, monkeypatch) -> None:
    wav = tmp_path / "final.wav"
    _write_wav(wav)
    monkeypatch.setattr(audio, "STORY_AAC_ARGS", ["-c:a", "no_such_codec"])
    assert audio.compress_story_wav(wav).name == "final.flac"


def test_compress_story_wav_keeps_wav_without_ffmpeg(tmp_path, monkeypatch) -> None:
    wav = tmp_path / "final.wav"
    _write_wav(wav)
    monkeypatch.setattr(audio, "_ffmpeg_bin", lambda: None)
    assert audio.compress_story_wav(wav) == wav


def test_story_ready_from_database_without_local_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(story_stream, "TTS_DIR", tmp_path)
    monkeypatch.setattr(story_stream.storage, "storage_configured", lambda: True)

    async def ready_key(story_key):
        return {"inbucket": "stories/inbucket.m4a", "local": "story_streams/local/final.wav"}.get(story_key)

    monkeypatch.setattr(story_stream.story_audio_store, "get_ready_key", ready_key)
    assert asyncio.run(story_stream.story_ready("inbucket")) is True
    # A local final_key whose file is gone is not ready: the story is made again.
    assert asyncio.run(story_stream.story_ready("local")) is False
    assert asyncio.run(story_stream.story_ready("unknown")) is False


def test_cleanup_removes_only_stored_and_idle_stories(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(story_cleanup, "TTS_DIR", tmp_path)
    monkeypatch.setattr(story_cleanup.storage, "storage_configured", lambda: True)
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    recent = datetime.now(timezone.utc) - timedelta(minutes=5)
    rows = {
        "done": ("stories/done.m4a", old),
        "recent": ("stories/recent.m4a", recent),
        "localkey": ("story_streams/localkey/final.wav", old),
        "missing": ("stories/missing.m4a", old),
        "playing": ("stories/playing.m4a", old),
    }
    for key in [*rows, "partial"]:
        _write_wav(tmp_path / "story_streams" / key / "chunk_0.wav")

    async def ready_rows(keys):
        return [(key, *rows[key]) for key in keys if key in rows]

    monkeypatch.setattr(story_cleanup.story_audio_store, "ready_rows", ready_rows)
    monkeypatch.setattr(story_cleanup.story_audio_store, "partial_rows", _no_partial_rows)
    monkeypatch.setattr(story_cleanup.storage, "exists", lambda key: key != "stories/missing.m4a")
    monkeypatch.setattr(story_cleanup, "story_busy", lambda key: key == "playing")
    monkeypatch.setattr(story_cleanup, "story_on_device", lambda key: False)

    assert asyncio.run(story_cleanup.clean_story_files()) == 1
    left = {path.name for path in (tmp_path / "story_streams").iterdir()}
    assert left == {"recent", "localkey", "missing", "playing", "partial"}


def test_cleanup_keeps_story_its_device_is_still_on(tmp_path, monkeypatch) -> None:
    # Paused for 2 hours on a saved story: kept. A day without playing it: removed.
    monkeypatch.setattr(story_cleanup, "TTS_DIR", tmp_path)
    monkeypatch.setattr(story_cleanup.storage, "storage_configured", lambda: True)
    now = datetime.now(timezone.utc)
    rows = {
        "paused": ("stories/paused.m4a", now - timedelta(hours=2)),
        "abandoned": ("stories/abandoned.m4a", now - timedelta(hours=25)),
    }
    for key in rows:
        _write_wav(tmp_path / "story_streams" / key / "chunk_0.wav")

    async def ready_rows(keys):
        return [(key, *rows[key]) for key in keys if key in rows]

    monkeypatch.setattr(story_cleanup.story_audio_store, "ready_rows", ready_rows)
    monkeypatch.setattr(story_cleanup.story_audio_store, "partial_rows", _no_partial_rows)
    monkeypatch.setattr(story_cleanup.storage, "exists", lambda key: True)
    monkeypatch.setattr(story_cleanup, "story_busy", lambda key: False)
    monkeypatch.setattr(story_cleanup, "story_on_device", lambda key: True)

    assert asyncio.run(story_cleanup.clean_story_files()) == 1
    assert {path.name for path in (tmp_path / "story_streams").iterdir()} == {"paused"}


def test_remove_left_story_only_when_saved_in_bucket(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(story_cleanup, "TTS_DIR", tmp_path)
    monkeypatch.setattr(story_cleanup.storage, "storage_configured", lambda: True)
    keys = {"saved": "stories/saved.m4a", "notuploaded": "stories/notuploaded.m4a", "localkey": "story_streams/x.wav"}
    for key in [*keys, "unfinished", "otherphone"]:
        _write_wav(tmp_path / "story_streams" / key / "chunk_0.wav")
    keys["otherphone"] = "stories/otherphone.m4a"
    old = datetime.now(timezone.utc)

    async def ready_rows(story_keys):
        return [(key, keys[key], old) for key in story_keys if key in keys]

    monkeypatch.setattr(story_cleanup.story_audio_store, "ready_rows", ready_rows)
    monkeypatch.setattr(story_cleanup.storage, "exists", lambda key: key != "stories/notuploaded.m4a")
    monkeypatch.setattr(story_cleanup, "story_in_use", lambda key: key == "otherphone")

    removed = {key for key in [*keys, "unfinished"] if asyncio.run(story_cleanup.remove_left_story(key))}
    assert removed == {"saved"}
    left = {path.name for path in (tmp_path / "story_streams").iterdir()}
    assert left == {"notuploaded", "localkey", "unfinished", "otherphone"}


def test_chunk_wait_answers_final_once_the_story_was_cleaned_up(tmp_path, monkeypatch) -> None:
    from app.api import tts

    monkeypatch.setattr(story_stream, "TTS_DIR", tmp_path)

    async def ready(story_key):
        return story_key == "saved"

    monkeypatch.setattr(tts, "story_ready", ready)
    assert asyncio.run(tts.wait_story_stream_chunk("saved", 3)) == {"status": "final", "detail": None}


def _fake_bucket_and_rows(tmp_path, monkeypatch) -> tuple[dict, dict]:
    """A bucket in tmp_path/bucket and the partial columns of story_audios in a dict."""
    bucket_dir = tmp_path / "bucket"
    bucket: dict[str, object] = {}
    rows: dict[str, tuple[str, list[int]]] = {}

    def upload_file(path, key):
        dest = bucket_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        bucket[key] = dest

    def download_file(key, path):
        if key not in bucket:
            raise FileNotFoundError(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bucket[key], path)

    async def set_partial(story_key, key, samples):
        rows[story_key] = (key, samples)

    async def get_partial(story_key):
        return rows.get(story_key)

    async def clear_partial(story_key):
        rows.pop(story_key, None)

    monkeypatch.setattr(story_partial.storage, "storage_configured", lambda: True)
    monkeypatch.setattr(story_partial.storage, "upload_file", upload_file)
    monkeypatch.setattr(story_partial.storage, "download_file", download_file)
    monkeypatch.setattr(story_partial.storage, "delete_file", lambda key: bucket.pop(key, None))
    monkeypatch.setattr(story_partial.story_audio_store, "set_partial", set_partial)
    monkeypatch.setattr(story_partial.story_audio_store, "get_partial", get_partial)
    monkeypatch.setattr(story_partial.story_audio_store, "clear_partial", clear_partial)
    return bucket, rows


@needs_ffmpeg
def test_pack_and_restore_give_back_the_same_chunks(tmp_path, monkeypatch) -> None:
    bucket, rows = _fake_bucket_and_rows(tmp_path, monkeypatch)
    folder = tmp_path / "story_streams" / "left"
    sizes = [24000, 30011, 17]
    for index, frames in enumerate(sizes):
        _write_noise_wav(folder / f"chunk_{index}.wav", frames, index)
    original = [_frames(folder / f"chunk_{index}.wav") for index in range(len(sizes))]

    assert asyncio.run(story_partial.pack_left_story("left", folder, lambda key: False)) is True
    assert not folder.exists()
    assert list(bucket) == ["tmp/left/partial.flac"]
    assert rows["left"] == ("tmp/left/partial.flac", sizes)

    assert asyncio.run(story_partial.restore_partial("left", folder)) == len(sizes)
    assert [_frames(folder / f"chunk_{index}.wav") for index in range(len(sizes))] == original
    assert sorted(path.name for path in folder.iterdir()) == ["chunk_0.wav", "chunk_1.wav", "chunk_2.wav"]
    # Restored: the bucket copy and the row fields are gone.
    assert not bucket and not rows


@needs_ffmpeg
def test_pack_stops_at_the_first_missing_chunk(tmp_path) -> None:
    folder = tmp_path / "gap"
    for index in (0, 1, 3):
        _write_noise_wav(folder / f"chunk_{index}.wav", 1000, index)
    flac, samples = story_partial.pack(folder)
    assert flac.name == "partial.flac" and samples == [1000, 1000]
    assert story_partial.pack(tmp_path / "empty") is None


def test_pack_skips_stories_in_use(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(story_cleanup, "TTS_DIR", tmp_path)
    monkeypatch.setattr(story_cleanup.storage, "storage_configured", lambda: True)
    now = datetime.now(timezone.utc)
    last_used = {
        "left": now - timedelta(hours=1),
        "recent": now - timedelta(minutes=5),
        "busy": now - timedelta(hours=1),
        "ondevice": now - timedelta(hours=1),
    }
    for key in last_used:
        _write_wav(tmp_path / "story_streams" / key / "chunk_0.wav")
    packed = []

    async def ready_rows(keys):
        return []

    async def partial_rows(keys):
        return [(key, last_used[key]) for key in keys if key in last_used]

    async def pack_left_story(story_key, folder, busy):
        packed.append(story_key)
        return True

    monkeypatch.setattr(story_cleanup.story_audio_store, "ready_rows", ready_rows)
    monkeypatch.setattr(story_cleanup.story_audio_store, "partial_rows", partial_rows)
    monkeypatch.setattr(story_cleanup.story_partial, "pack_left_story", pack_left_story)
    monkeypatch.setattr(story_cleanup, "story_busy", lambda key: key == "busy")
    monkeypatch.setattr(story_cleanup, "story_on_device", lambda key: key == "ondevice")

    assert asyncio.run(story_cleanup.clean_story_files()) == 1
    assert packed == ["left"]


def test_pack_keeps_folder_when_played_again_meanwhile(tmp_path, monkeypatch) -> None:
    bucket, rows = _fake_bucket_and_rows(tmp_path, monkeypatch)
    folder = tmp_path / "back"
    _write_wav(folder / "chunk_0.wav")
    flac = folder / "partial.flac"
    monkeypatch.setattr(story_partial, "pack", lambda path: (flac.write_bytes(b"x") and flac, [24000]))
    checks = iter([False, True])

    assert asyncio.run(story_partial.pack_left_story("back", folder, lambda key: next(checks))) is False
    assert (folder / "chunk_0.wav").exists()
    assert not bucket and not rows


def test_missing_bucket_partial_clears_the_row(tmp_path, monkeypatch) -> None:
    bucket, rows = _fake_bucket_and_rows(tmp_path, monkeypatch)
    rows["gone"] = ("tmp/gone/partial.flac", [24000])
    folder = tmp_path / "gone"
    assert asyncio.run(story_partial.restore_partial("gone", folder)) == 0
    assert not rows
    assert not (folder / "chunk_0.wav").exists()
