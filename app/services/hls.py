"""HLS for a story while it is generated: one playlist URL that grows chunk by chunk and plays
without a gap at chunk changes.

Each chunk_N.wav becomes seg_N.m4s (FLAC in fragmented MP4) the first time it is asked for, cached
next to the wav. FLAC because each chunk is encoded on its own: AAC would add ~85 ms of encoder
delay to every segment, which is the gap again. FLAC is lossless and adds nothing.

The player's timeline comes from each segment's tfdt box (its start in samples). ffmpeg writes 0
there, so the exact start (the frames of all earlier chunks) is written in afterwards. One init.mp4
serves every segment: the FLAC header fields that differ per chunk (frame sizes, total samples,
MD5) are set to 0, which FLAC defines as "unknown".

Works on a story folder only (no import of story_stream), so story_stream can use it.
"""

import asyncio
import os
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

from app.services.audio import _ffmpeg_bin

# Fixed for a growing playlist. MAX_WORDS = 50 in story_chunker gives chunks of ~20–25 s.
HLS_TARGET_S = 5
# "-strict experimental": some ffmpeg builds only write FLAC into MP4 with it.
HLS_SEGMENT_CODEC = ["-c:a", "flac", "-strict", "experimental"]
INIT_NAME = "init.mp4"
PLAYLIST_TYPE = "application/vnd.apple.mpegurl"
SEGMENT_TYPE = "audio/mp4"

# (folder, index) -> lock, so two requests never encode the same segment at once.
_locks: dict[tuple[str, int], asyncio.Lock] = {}


def chunk_path(folder: Path, index: int) -> Path:
    return folder / f"chunk_{index}.wav"


def segment_path(folder: Path, index: int) -> Path:
    return folder / f"seg_{index}.m4s"


def init_path(folder: Path) -> Path:
    return folder / INIT_NAME


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def made_count(folder: Path) -> int:
    """Chunks made from 0 on without a gap: only these can be played in order."""
    count = 0
    while _exists(chunk_path(folder, count)):
        count += 1
    return count


def _chunk_frames(folder: Path, index: int) -> tuple[int, int]:
    """(frames, sample rate) from the chunk's WAV header."""
    with wave.open(str(chunk_path(folder, index)), "rb") as src:
        return src.getnframes(), src.getframerate()


def made_frames(folder: Path) -> tuple[list[int], int]:
    """Frames of every made chunk (no gap) and the sample rate (24000 if nothing is made yet)."""
    frames: list[int] = []
    rate = 24000
    for index in range(made_count(folder)):
        count, rate = _chunk_frames(folder, index)
        frames.append(count)
    return frames, rate


def chunk_at(folder: Path, position_ms: int) -> tuple[int, int] | None:
    """(chunk index, its start in ms) that holds `position_ms`, or None past the made chunks."""
    frames, rate = made_frames(folder)
    start = 0
    for index, count in enumerate(frames):
        if position_ms * rate < (start + count) * 1000:
            return index, start * 1000 // rate
        start += count
    return None


def made_ms(folder: Path) -> int:
    """Length of the made chunks: the part of the story that can be played and seeked into."""
    frames, rate = made_frames(folder)
    return sum(frames) * 1000 // rate


def build_playlist(folder: Path, ended: bool) -> str:
    """The playlist for the made chunks. `ended` (story ready, or failed) closes it; until then the
    player reloads it and plays new segments as they appear."""
    frames, rate = made_frames(folder)
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:7",
        "#EXT-X-PLAYLIST-TYPE:EVENT",
        f"#EXT-X-TARGETDURATION:{HLS_TARGET_S}",
        # A growing playlist would otherwise start near its end, not at the story's start.
        "#EXT-X-START:TIME-OFFSET=0,PRECISE=YES",
        f'#EXT-X-MAP:URI="{INIT_NAME}"',
    ]
    for index, count in enumerate(frames):
        seconds = count / rate
        if seconds > HLS_TARGET_S:
            print(f"[hls] {folder.name[:8]}: chunk {index} is {seconds:.1f}s, over {HLS_TARGET_S}s", flush=True)
        lines += [f"#EXTINF:{seconds:.3f},", f"seg/{index}.m4s"]
    if ended:
        lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines) + "\n"


def _top_boxes(data: bytes) -> list[tuple[str, int, int]]:
    """Top-level MP4 boxes as (type, offset, size)."""
    boxes: list[tuple[str, int, int]] = []
    at = 0
    while at + 8 <= len(data):
        size, kind = struct.unpack(">I4s", data[at:at + 8])
        if size < 8:
            raise ValueError(f"bad MP4 box at {at}")
        boxes.append((kind.decode("latin1"), at, size))
        at += size
    return boxes


def _box_at(data: bytes | bytearray, kind: bytes) -> int:
    at = data.find(kind) - 4
    if at < 0:
        raise ValueError(f"no {kind.decode()} box")
    return at


def shared_init(init: bytes) -> bytes:
    """The init segment with the per-chunk FLAC STREAMINFO fields set to 0 (= unknown), so it is the
    same for every chunk of the same format."""
    data = bytearray(init)
    # dfLa box: size(4) type(4) version/flags(4) metadata block header(4), then STREAMINFO (34 bytes).
    info = _box_at(data, b"dfLa") + 16
    data[info + 4:info + 10] = bytes(6)  # min and max frame size (24 bits each)
    # Bits 108–143: the last 4 bits of byte 13 (bits/sample) stay, total samples (36 bits) become 0.
    data[info + 13] &= 0xF0
    data[info + 14:info + 18] = bytes(4)
    data[info + 18:info + 34] = bytes(16)  # MD5 of the audio
    return bytes(data)


def place_segment(segment: bytes, index: int, start_frames: int) -> bytes:
    """Puts a segment at its place in the story: drops the sidx (optional in HLS, and its time would
    be wrong), sets the fragment number and the start (tfdt, in samples) to `start_frames`."""
    data = bytearray(b"".join(segment[at:at + size] for kind, at, size in _top_boxes(segment) if kind != "sidx"))
    mfhd = _box_at(data, b"mfhd")
    data[mfhd + 12:mfhd + 16] = struct.pack(">I", index + 1)
    tfdt = _box_at(data, b"tfdt")
    if data[tfdt + 8] == 1:
        data[tfdt + 12:tfdt + 20] = struct.pack(">Q", start_frames)
    elif start_frames <= 0xFFFFFFFF:
        data[tfdt + 12:tfdt + 16] = struct.pack(">I", start_frames)
    else:
        raise ValueError("story too long for a 32-bit tfdt")
    return bytes(data)


def _write_atomic(path: Path, data: bytes) -> None:
    # Through a temp file, so a reader never sees half a segment.
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def encode_segment(folder: Path, index: int) -> Path:
    """Writes seg_N.m4s (and init.mp4 if missing) from chunk_N.wav. Blocks: run it in a thread."""
    if index >= made_count(folder):
        raise FileNotFoundError(f"chunk {index} or an earlier one is not made yet")
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg missing")
    start = sum(_chunk_frames(folder, earlier)[0] for earlier in range(index))
    work = Path(tempfile.mkdtemp(dir=folder, prefix=f".hls_{index}_"))
    try:
        # Relative names in its own folder: the HLS muxer wants a %d pattern for segment names.
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-i", str(chunk_path(folder, index).resolve()),
                *HLS_SEGMENT_CODEC,
                "-f", "hls", "-hls_segment_type", "fmp4", "-hls_time", "100000",
                "-hls_playlist_type", "vod", "-start_number", str(index),
                "-hls_fmp4_init_filename", INIT_NAME, "-hls_segment_filename", "seg_%d.m4s", "out.m3u8",
            ],
            cwd=work,
            check=True,
            capture_output=True,
        )
        segment = place_segment((work / f"seg_{index}.m4s").read_bytes(), index, start)
        if not _exists(init_path(folder)):
            _write_atomic(init_path(folder), shared_init((work / INIT_NAME).read_bytes()))
        _write_atomic(segment_path(folder, index), segment)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return segment_path(folder, index)


async def encode_ready(folder: Path) -> None:
    """Encodes every made chunk (no gap) that has no segment yet, so the player never waits for
    ffmpeg. Called right after a chunk is written; errors are only logged (the request path still
    encodes on demand)."""
    for index in range(made_count(folder)):
        if _exists(segment_path(folder, index)) and _exists(init_path(folder)):
            continue
        try:
            await ensure_segment(folder, index)
        except Exception as error:  # noqa: BLE001 — a background pre-encode must never crash
            print(f"[hls] {folder.name[:8]}: pre-encode of segment {index} failed: {error}", flush=True)
            return


async def ensure_segment(folder: Path, index: int) -> Path:
    """seg_N.m4s, encoded once on first use. FileNotFoundError while the chunk (or an earlier one)
    is not made, RuntimeError if ffmpeg is missing or fails."""
    path = segment_path(folder, index)
    if _exists(path) and _exists(init_path(folder)):
        return path
    key = (os.fspath(folder), index)
    lock = _locks.setdefault(key, asyncio.Lock())
    try:
        async with lock:
            if _exists(path) and _exists(init_path(folder)):
                return path
            try:
                return await asyncio.to_thread(encode_segment, folder, index)
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or b"").decode("utf-8", "replace").strip()[-300:]
                raise RuntimeError(f"ffmpeg could not make segment {index}: {detail}") from error
    finally:
        if not lock.locked() and _locks.get(key) is lock:
            _locks.pop(key, None)
