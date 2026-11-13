"""Step 0 of the HLS change: check FLAC-in-fMP4 HLS on real phones before relying on it.

Builds init.mp4, seg_N.m4s and playlist.m3u8 for the chunks of one story folder (same code as the
server), checks the joined audio is sample-identical to the chunks, and serves the folder so a phone
on the same network can play http://<pc-ip>:<port>/playlist.m3u8 in the app's dev client.

Usage (from suno-tts/, venv active):
    python scripts/hls_spike.py <story key or folder> [--out DIR] [--serve 8090]
"""

import argparse
import functools
import http.server
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import hls  # noqa: E402
from app.services.audio import _ffmpeg_bin  # noqa: E402
from app.store import TTS_DIR  # noqa: E402


def build(source: Path, out: Path) -> int:
    count = hls.made_count(source)
    if not count:
        raise SystemExit(f"No chunk_0.wav in {source}")
    out.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        shutil.copy(hls.chunk_path(source, index), hls.chunk_path(out, index))
        hls.encode_segment(out, index)
    (out / "seg").mkdir(exist_ok=True)
    for index in range(count):
        hls.segment_path(out, index).replace(out / "seg" / f"{index}.m4s")
    (out / "playlist.m3u8").write_text(hls.build_playlist(out, ended=True), encoding="utf-8")
    return count


def check_identical(out: Path, count: int) -> bool:
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise SystemExit("ffmpeg missing")
    with tempfile.TemporaryDirectory() as tmp:
        decoded = Path(tmp) / "decoded.wav"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-allowed_extensions", "ALL",
             "-i", str(out / "playlist.m3u8"), "-c:a", "pcm_s16le", str(decoded)],
            check=True,
        )
        original = b""
        for index in range(count):
            with wave.open(str(hls.chunk_path(out, index)), "rb") as src:
                original += src.readframes(src.getnframes())
        with wave.open(str(decoded), "rb") as src:
            joined = src.readframes(src.getnframes())
    print(f"chunks: {len(original) // 2} frames, HLS decoded: {len(joined) // 2} frames")
    return joined == original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("story", help="story key (folder in data/tts/story_streams) or a folder path")
    parser.add_argument("--out", help="output folder (default: data/tts/hls_spike/<key>)")
    parser.add_argument("--serve", type=int, default=0, help="serve the output folder on this port")
    args = parser.parse_args()

    source = Path(args.story)
    if not source.is_dir():
        source = TTS_DIR / "story_streams" / args.story
    out = Path(args.out) if args.out else TTS_DIR / "hls_spike" / source.name
    count = build(source, out)
    same = check_identical(out, count)
    print(f"{count} segments in {out}: {'IDENTICAL to the chunks (gapless)' if same else 'NOT identical'}")
    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
        handler.extensions_map.update({".m3u8": hls.PLAYLIST_TYPE, ".m4s": hls.SEGMENT_TYPE, ".mp4": hls.SEGMENT_TYPE})
        print(f"Serving on http://0.0.0.0:{args.serve}/playlist.m3u8 (Ctrl+C stops)")
        http.server.ThreadingHTTPServer(("0.0.0.0", args.serve), handler).serve_forever()


if __name__ == "__main__":
    main()
