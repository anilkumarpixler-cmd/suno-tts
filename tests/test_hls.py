import asyncio
import math
import subprocess
import wave
from array import array
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api import tts as tts_api
from app.services import hls, story_stream
from app.services.audio import _ffmpeg_bin

RATE = 24000
# Chunk lengths in frames: odd sizes, so a wrong offset can't line up by chance.
FRAMES = [RATE + 123, RATE // 2 + 7, RATE * 2 - 31]

needs_ffmpeg = pytest.mark.skipif(not _ffmpeg_bin(), reason="ffmpeg missing")


def _write_chunk(folder: Path, index: int, frames: int) -> None:
    samples = array("h", (int(8000 * math.sin(2 * math.pi * (220 + 40 * index) * i / RATE)) for i in range(frames)))
    with wave.open(str(folder / f"chunk_{index}.wav"), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(RATE)
        dst.writeframes(samples.tobytes())


def _story(folder: Path, indexes=range(len(FRAMES))) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for index in indexes:
        _write_chunk(folder, index, FRAMES[index])
    return folder


def test_playlist_lists_only_chunks_without_a_gap(tmp_path) -> None:
    folder = _story(tmp_path / "s", indexes=[0, 1])
    _write_chunk(folder, 3, RATE)  # chunk 2 missing: 3 can't be played in order yet
    playlist = hls.build_playlist(folder, ended=False)
    assert "seg/0.m4s" in playlist and "seg/1.m4s" in playlist
    assert "seg/3.m4s" not in playlist
    assert f"#EXTINF:{FRAMES[0] / RATE:.3f}," in playlist
    assert "#EXT-X-START:TIME-OFFSET=0,PRECISE=YES" in playlist
    assert "#EXT-X-PLAYLIST-TYPE:EVENT" in playlist
    assert "#EXT-X-ENDLIST" not in playlist


def test_playlist_is_closed_once_ended(tmp_path) -> None:
    folder = _story(tmp_path / "s")
    assert hls.build_playlist(folder, ended=True).rstrip().endswith("#EXT-X-ENDLIST")


def test_chunk_at_finds_the_chunk_and_its_start(tmp_path) -> None:
    folder = _story(tmp_path / "s")
    second_start_ms = FRAMES[0] * 1000 // RATE
    assert hls.chunk_at(folder, 0) == (0, 0)
    assert hls.chunk_at(folder, second_start_ms + 100) == (1, second_start_ms)
    assert hls.chunk_at(folder, 60_000) is None
    assert hls.made_ms(folder) == sum(FRAMES) * 1000 // RATE


@needs_ffmpeg
def test_segments_join_sample_exact(tmp_path) -> None:
    # The whole point of HLS here: the joined segments are exactly the chunks, no gap, no overlap.
    folder = _story(tmp_path / "s")
    for index in range(len(FRAMES)):
        hls.encode_segment(folder, index)
    playlist = folder / "playlist.m3u8"
    playlist.write_text(hls.build_playlist(folder, ended=True).replace("seg/", "seg_"), encoding="utf-8")
    decoded = tmp_path / "decoded.wav"
    subprocess.run(
        [_ffmpeg_bin(), "-y", "-loglevel", "error", "-allowed_extensions", "ALL", "-i", str(playlist),
         "-c:a", "pcm_s16le", str(decoded)],
        check=True,
    )
    original = b""
    for index in range(len(FRAMES)):
        with wave.open(str(folder / f"chunk_{index}.wav"), "rb") as src:
            original += src.readframes(src.getnframes())
    with wave.open(str(decoded), "rb") as src:
        assert src.readframes(src.getnframes()) == original


@needs_ffmpeg
def test_segments_share_one_init(tmp_path) -> None:
    # Chunks of different length must not need different init segments.
    one = _story(tmp_path / "a", indexes=[0])
    two = _story(tmp_path / "b", indexes=[0, 1, 2])
    hls.encode_segment(one, 0)
    hls.encode_segment(two, 2)
    assert hls.init_path(one).read_bytes() == hls.init_path(two).read_bytes()


def test_segment_needs_every_earlier_chunk(tmp_path) -> None:
    folder = _story(tmp_path / "s", indexes=[0])
    _write_chunk(folder, 2, RATE)
    with pytest.raises(FileNotFoundError):
        hls.encode_segment(folder, 2)


def test_resume_point(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(story_stream, "TTS_DIR", tmp_path)
    _story(story_stream.story_stream_dir("k"), indexes=[0, 1])
    second_start_ms = FRAMES[0] * 1000 // RATE
    assert story_stream.resume_point("k", 0) == (0, 0, False)
    assert story_stream.resume_point("k", second_start_ms + 50) == (1, second_start_ms, False)
    # Past the made chunks (a restore failed): the first chunk not made yet, and the app is told.
    made_ms = (FRAMES[0] + FRAMES[1]) * 1000 // RATE
    assert story_stream.resume_point("k", made_ms + 5_000) == (2, made_ms, True)


def _routes(tmp_path, monkeypatch, ready: bool = False) -> Path:
    folder = _story(tmp_path / "streams" / "k", indexes=[0, 1])
    monkeypatch.setattr(tts_api, "story_stream_dir", lambda key: tmp_path / "streams" / key)
    monkeypatch.setattr(tts_api, "story_stream_error", lambda key: None)

    async def is_ready(key, check_bucket=False):
        return ready

    async def alive(key):
        return None

    monkeypatch.setattr(tts_api, "story_ready", is_ready)
    monkeypatch.setattr(tts_api, "keep_stream_alive", alive)
    monkeypatch.setattr(tts_api, "note_listening", lambda key, index: None)
    monkeypatch.setattr(tts_api, "story_busy", lambda key: True)
    return folder


def test_playlist_route(tmp_path, monkeypatch) -> None:
    _routes(tmp_path, monkeypatch)
    response = asyncio.run(tts_api.get_story_playlist("k"))
    assert response.media_type == hls.PLAYLIST_TYPE
    assert response.headers["cache-control"] == "no-cache"
    assert b"seg/1.m4s" in response.body and b"#EXT-X-ENDLIST" not in response.body
    with pytest.raises(HTTPException) as missing:
        asyncio.run(tts_api.get_story_playlist("unknown"))
    assert missing.value.status_code == 404


def test_segment_route_404_until_made(tmp_path, monkeypatch) -> None:
    _routes(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as missing:
        asyncio.run(tts_api.get_story_segment("k", 2))
    assert missing.value.status_code == 404


@needs_ffmpeg
def test_segment_route_serves_cached_segment(tmp_path, monkeypatch) -> None:
    folder = _routes(tmp_path, monkeypatch)
    response = asyncio.run(tts_api.get_story_segment("k", 1))
    assert Path(response.path) == hls.segment_path(folder, 1)
    assert "immutable" in response.headers["cache-control"]
    assert hls.init_path(folder).exists()


def test_pause_by_position_finds_the_chunk(tmp_path, monkeypatch) -> None:
    _routes(tmp_path, monkeypatch)
    paused: list[int] = []
    monkeypatch.setattr(tts_api, "pause_story_stream", lambda key, chunk: paused.append(chunk) or True)
    middle_of_second = FRAMES[0] * 1000 // RATE + 100
    asyncio.run(tts_api.pause_story_stream_route("k", position_ms=middle_of_second))
    asyncio.run(tts_api.pause_story_stream_route("k", chunk=3))  # the old app still sends a chunk
    asyncio.run(tts_api.pause_story_stream_route("k", position_ms=999_999))  # past the made chunks
    assert paused == [1, 3, 2]


def test_status_route(tmp_path, monkeypatch) -> None:
    _routes(tmp_path, monkeypatch)
    status = asyncio.run(tts_api.get_story_status("k"))
    assert status["status"] == "generating"
    assert status["made_chunks"] == 2
    assert status["made_ms"] == (FRAMES[0] + FRAMES[1]) * 1000 // RATE
