import audioop
from array import array
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import wave

from fastapi import UploadFile
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

from app.store import RECORDINGS_DIR, TTS_DIR, ensure_dirs

WAV_EXTENSIONS = {".wav"}

# Bump when prompt/chunk processing changes so cached story audio is regenerated.
AUDIO_PIPELINE_VERSION = "2"

# Chatterbox clones the prompt's room noise along with the voice. Removing rumble, the
# ~100 Hz mains hum and steady hiss (and trimming the ends) cut the noisy lead-in at the
# start of each generated chunk from 90-200 ms to ~20 ms in A/B tests on Colab.
PROMPT_FILTER = (   
    "highpass=f=70,"
    "bandreject=f=100:width_type=h:w=12,"
    "afftdn=nr=12:nf=-45,"
    "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
    "silenceremove=start_periods=1:start_threshold=-45dB,areverse"
)

# Silence appended to every chunk except the last, so a chunk change sounds like a full stop.
CHUNK_PAUSE_MS = 0
_FRAME_MS = 10
_SPEECH_DROP = 10 ** (17 / 20)  # cut level = 90th-percentile frame RMS - 17 dB (tuned on real chunks)
_ONSET_FRAMES = 6  # speech must hold for most of 60 ms, so a single noise blip is skipped
_PRE_ROLL_MS = 2
_POST_ROLL_MS = 80
_FADE_IN_MS = 20  # spans the pre-roll, so what is left of the lead-in ramps up from silence
_FADE_OUT_MS =50
_MIN_CLEAN_S = 0.3


def _extension(filename: str | None) -> str:
    if not filename or "." not in filename:
        return "m4a"
    return filename.rsplit(".", 1)[-1].lower()[:8]


def save_bytes(voice_id: str, data: bytes, filename: str | None) -> Path:
    ensure_dirs()
    dest = RECORDINGS_DIR / f"{voice_id}.{_extension(filename)}"
    dest.write_bytes(data)
    wav = ensure_wav(dest)
    return wav if wav.suffix.lower() == ".wav" else dest


async def save_recording(voice_id: str, upload: UploadFile) -> Path:
    return save_bytes(voice_id, await upload.read(), upload.filename)


def delete_file(path: str | None) -> None:
    if not path:
        return
    file = Path(path)
    if file.exists():
        file.unlink()


def tts_path(tts_id: str, ext: str = "mp3") -> Path:
    ensure_dirs()
    suffix = ext if ext.startswith(".") else f".{ext}"
    return TTS_DIR / f"{tts_id}{suffix}"


def find_tts_path(tts_id: str) -> Path | None:
    ensure_dirs()
    for suffix in (".wav", ".mp3"):
        path = TTS_DIR / f"{tts_id}{suffix}"
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def find_cloned_tts_path(tts_id: str) -> Path | None:
    ensure_dirs()
    path = TTS_DIR / f"{tts_id}.wav"
    if path.exists() and path.stat().st_size > 0:
        return path
    return None


def _speech_bounds(levels: list[int], cut: float) -> tuple[int, int] | None:
    """First and last frame where speech holds above `cut` for most of _ONSET_FRAMES."""
    need = _ONSET_FRAMES - 1
    count = len(levels)
    start = next(
        (i for i in range(count - _ONSET_FRAMES + 1) if sum(v > cut for v in levels[i:i + _ONSET_FRAMES]) >= need),
        None,
    )
    end = next(
        (i for i in range(count - 1, _ONSET_FRAMES - 2, -1)
         if sum(v > cut for v in levels[i - _ONSET_FRAMES + 1:i + 1]) >= need),
        None,
    )
    if start is None or end is None or end < start:
        return None
    return start, end


def clean_chunk_wav(data: bytes, pad_end_ms: int = 0) -> bytes:
    """Cut the noise Chatterbox leaves before and after speech, fade both ends to zero so chunk
    changes don't click, and append `pad_end_ms` of silence as the pause before the next chunk."""
    try:
        with wave.open(BytesIO(data), "rb") as src:
            params = src.getparams()
            raw = src.readframes(src.getnframes())
    except (wave.Error, EOFError):
        return data
    if params.sampwidth != 2 or params.nchannels != 1 or not raw:
        return data

    samples = array("h", raw)
    rate = params.framerate
    frame = rate * _FRAME_MS // 1000
    levels = [audioop.rms(raw[offset:offset + frame * 2], 2) for offset in range(0, len(raw), frame * 2)]
    speech_level = sorted(levels)[int(len(levels) * 0.9)]
    bounds = _speech_bounds(levels, speech_level / _SPEECH_DROP)
    if bounds is None:
        return data

    start = max(0, bounds[0] * frame - rate * _PRE_ROLL_MS // 1000)
    end = min(len(samples), (bounds[1] + 1) * frame + rate * _POST_ROLL_MS // 1000)
    if end - start < rate * _MIN_CLEAN_S:
        return data

    trimmed = samples[start:end]
    fade_in = min(len(trimmed), rate * _FADE_IN_MS // 1000)
    for i in range(fade_in):
        trimmed[i] = int(trimmed[i] * i / fade_in)
    fade_out = min(len(trimmed), rate * _FADE_OUT_MS // 1000)
    for i in range(fade_out):
        trimmed[-1 - i] = int(trimmed[-1 - i] * i / fade_out)
    trimmed.extend([0] * (rate * pad_end_ms // 1000))

    out = BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setparams(params)
        dst.writeframes(trimmed.tobytes())
    return out.getvalue()


def concat_wav_bytes(parts: list[bytes], dest: Path) -> Path:
    """Join chunks that clean_chunk_wav already trimmed and padded with their pause."""
    if not parts:
        raise RuntimeError("No wav chunks to stitch")

    frames: list[bytes] = []
    params = None

    for data in parts:
        with wave.open(BytesIO(data), "rb") as src:
            next_params = src.getparams()
            if params is None:
                params = next_params
            elif (src.getnchannels(), src.getsampwidth(), src.getframerate()) != (
                params.nchannels,
                params.sampwidth,
                params.framerate,
            ):
                raise RuntimeError("Colab wav chunks have mismatched format")

            frames.append(src.readframes(src.getnframes()))

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.wav")
    with wave.open(str(tmp), "wb") as out:
        assert params is not None
        out.setparams(params)
        for chunk in frames:
            out.writeframes(chunk)

    tmp.replace(dest)
    return dest


def media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix in {".m4a", ".mp4"}:
        return "audio/mp4"
    return "audio/mpeg"


def audio_duration(path: Path) -> float:
    try:
        if path.suffix.lower() == ".wav":
            return max(1.0, float(WAVE(path).info.length))
        return max(1.0, float(MP3(path).info.length))
    except Exception:
        return 1.0


def _ffmpeg_bin() -> str | None:
    for name in ("ffmpeg", "ffmpeg.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


def ensure_wav(path: Path) -> Path:
    if not path.exists():
        return path
    if path.suffix.lower() in WAV_EXTENSIONS:
        return path

    wav = path.with_suffix(".wav")
    if wav.exists() and wav.stat().st_size > 0:
        return wav

    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        return path

    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-ac", "1", "-ar", "24000", str(wav)],
            check=True,
            capture_output=True,
        )
        if wav.exists() and wav.stat().st_size > 0:
            return wav
    except Exception:
        wav.unlink(missing_ok=True)
    return path


def prepare_prompt_wav(source: Path) -> Path:
    """Cleaned copy of a voice recording for cloning (`<name>.prompt.wav`), rebuilt when the
    recording changes. Falls back to the original file if ffmpeg is missing or fails."""
    dest = source.with_name(f"{source.stem}.prompt.wav")
    if dest.exists() and dest.stat().st_size > 0 and dest.stat().st_mtime >= source.stat().st_mtime:
        return dest

    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        return source

    tmp = dest.with_suffix(".tmp.wav")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-af", PROMPT_FILTER,
             "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(tmp)],
            check=True,
            capture_output=True,
        )
        if tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(dest)
            return dest
    except Exception as error:
        print(f"[audio] prompt cleanup failed, using original: {error}", flush=True)
    tmp.unlink(missing_ok=True)
    return source
