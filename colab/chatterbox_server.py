"""Colab GPU FastAPI: clone story speech with Chatterbox Multilingual."""

from __future__ import annotations

import os
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from threading import Lock

# import numpy as np
# import noisereduce as nr

import torch
import torchaudio
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

SECRET = os.environ.get("COLAB_TTS_SECRET", "").strip()
 
MODEL = None
MODEL_LOCK = Lock()

app = FastAPI(title="Suno Chatterbox Colab", version="1.0.0")


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ChatterboxMultilingualTTS on {device}")
    # MODEL = ChatterboxMultilingualTTS.from_pretrained(device=device)
    MODEL = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
    return MODEL


def _check_secret(x_tts_secret: str | None) -> None:
    if SECRET and (x_tts_secret or "").strip() != SECRET:
        raise HTTPException(401, "Invalid secret")


def _language_id(value: str | None) -> str:
    text = (value or "en").strip().lower()
    if text in {"hi", "hindi", "hi-in"}:
        return "hi"
    return "en"

# def clean_audio(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
#     """
#     Light noise reduction for Chatterbox output.
#     Keeps the processing moderate so the voice does not become metallic.
#     """

#     # [channels, samples] -> mono numpy
#     audio = wav.detach().cpu()

#     if audio.dim() == 2:
#         audio_np = audio.mean(dim=0).numpy()
#     else:
#         audio_np = audio.numpy()

#     # Light noise reduction
#     cleaned = nr.reduce_noise(
#         y=audio_np,
#         sr=sample_rate,
#         stationary=False,
#         prop_decrease=1.5,
#     )

#     # Prevent clipping
#     peak = np.max(np.abs(cleaned))

#     if peak > 0.98:
#         cleaned = cleaned / peak * 0.98

#     return torch.from_numpy(cleaned).unsqueeze(0).float()
    

@app.get("/health")
def health():
    return {"ok": True, "cuda": torch.cuda.is_available(), "loaded": MODEL is not None}


@app.post("/generate")
async def generate(
    text: str = Form(...),
    language_id: str = Form("en"),
    audio: UploadFile = File(...),
    x_tts_secret: str | None = Header(default=None),
):
    _check_secret(x_tts_secret)
    story = text.strip()
    if not story:
        raise HTTPException(400, "Text is required")

    suffix = Path(audio.filename or "prompt.wav").suffix or ".wav"
    prompt = Path(tempfile.gettempdir()) / f"suno_prompt_{uuid.uuid4().hex}{suffix}"
    prompt.write_bytes(await audio.read())
    if prompt.stat().st_size == 0:
        prompt.unlink(missing_ok=True)
        raise HTTPException(400, "Audio file is required")

    model = load_model()
    lang = _language_id(language_id)


    try:

        with MODEL_LOCK:
            wav, sample_rate = model.synthesize(story, lang, str(prompt))
            # wav = clean_audio(wav, sample_rate)
            buffer = BytesIO()
            torchaudio.save(buffer, wav, sample_rate, format="wav")
            buffer.seek(0)
            return Response(content=buffer.read(), media_type="audio/wav")

    finally:
        prompt.unlink(missing_ok=True)
