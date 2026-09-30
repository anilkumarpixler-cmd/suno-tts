import math
import random
import wave
from array import array
from io import BytesIO

from app.services.audio import clean_chunk_wav, concat_wav_bytes

RATE = 24000


def _level(db: float) -> float:
    return 32768 * 10 ** (db / 20)


def _noise(seconds: float, db: float) -> list[int]:
    rng = random.Random(1)
    amp = _level(db) * math.sqrt(3)  # uniform noise RMS = amp / sqrt(3)
    return [int(rng.uniform(-amp, amp)) for _ in range(int(RATE * seconds))]


def _tone(seconds: float, db: float) -> list[int]:
    amp = _level(db) * math.sqrt(2)
    return [int(amp * math.sin(2 * math.pi * 220 * i / RATE)) for i in range(int(RATE * seconds))]


def _wav(samples: list[int]) -> bytes:
    out = BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(RATE)
        dst.writeframes(array("h", samples).tobytes())
    return out.getvalue()


def _samples(data: bytes) -> array:
    with wave.open(BytesIO(data), "rb") as src:
        return array("h", src.readframes(src.getnframes()))


def _rms_db(samples) -> float:
    if not samples:
        return -999.0
    return 20 * math.log10(max(1e-9, math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768))


def _chunk() -> bytes:
    # Chatterbox-like chunk: noisy lead-in, speech, noisy tail.
    return _wav(_noise(0.2, -42) + _tone(1.0, -18) + _noise(0.2, -42))


def test_clean_removes_noisy_lead_in_and_tail() -> None:
    cleaned = _samples(clean_chunk_wav(_chunk()))
    seconds = len(cleaned) / RATE
    # 1 s of speech plus at most the 20 ms pre-roll and 80 ms post-roll.
    assert 1.0 <= seconds <= 1.12
    # Only the 20 ms faded pre-roll is left before the tone, quieter than the old -42 dB lead-in.
    assert _rms_db(cleaned[: RATE * 20 // 1000]) < -45


def test_clean_fades_to_zero_at_both_ends() -> None:
    cleaned = _samples(clean_chunk_wav(_chunk()))
    assert cleaned[0] == 0
    assert cleaned[-1] == 0


def test_clean_pads_the_full_stop_pause() -> None:
    plain = _samples(clean_chunk_wav(_chunk()))
    padded = _samples(clean_chunk_wav(_chunk(), pad_end_ms=250))
    pause = RATE * 250 // 1000
    assert len(padded) == len(plain) + pause
    assert not any(padded[-pause:])


def test_clean_keeps_input_that_is_too_short_or_silent() -> None:
    short = _wav(_tone(0.1, -18))
    silent = _wav([0] * RATE)
    assert clean_chunk_wav(short) == short
    assert clean_chunk_wav(silent) == silent


def test_clean_ignores_non_wav_bytes() -> None:
    assert clean_chunk_wav(b"not audio") == b"not audio"


def test_concat_joins_chunks_without_extra_gap(tmp_path) -> None:
    first = clean_chunk_wav(_chunk(), pad_end_ms=250)
    last = clean_chunk_wav(_chunk())
    dest = concat_wav_bytes([first, last], tmp_path / "final.wav")
    assert len(_samples(dest.read_bytes())) == len(_samples(first)) + len(_samples(last))
