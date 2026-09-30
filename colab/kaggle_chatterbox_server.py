"""Kaggle GPU FastAPI: clone story speech with Chatterbox Multilingual on every GPU.

Paste into the Kaggle cell that starts with %%writefile /kaggle/working/chatterbox_server.py.
On "GPU T4 x2" one model is loaded per GPU (cuda:0 and cuda:1), so two chunks generate at once.
"""

from __future__ import annotations
import asyncio, contextlib, os, queue, tempfile, uuid, hashlib
from pathlib import Path
from threading import Lock
import torch, torchaudio
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
SECRET = os.environ.get("COLAB_TTS_SECRET", "").strip()
# One model per GPU: cuda:0 and cuda:1 on Kaggle "T4 x2".
MODELS = []
FREE_GPUS = queue.Queue()  # indexes of GPUs not busy right now
LOAD_LOCK = Lock()
# Prepared voices per GPU, keyed by (gpu index, hash of the reference WAV bytes).
VOICE_CACHE = {}
app = FastAPI(title="Suno Chatterbox Colab", version="1.0.0")
def load_models():
    with LOAD_LOCK:
        if MODELS:
            return MODELS
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
        for index, device in enumerate(devices):
            print("Loading ChatterboxMultilingualTTS on", device)
            MODELS.append(ChatterboxMultilingualTTS.from_pretrained(device=device))
            FREE_GPUS.put(index)
        return MODELS
def _check_secret(x_tts_secret):
    if SECRET and (x_tts_secret or "").strip() != SECRET:
        raise HTTPException(401, "Invalid secret")
def _language_id(value):
    text = (value or "en").strip().lower()
    return "hi" if text in {"hi", "hindi", "hi-in"} else "en"
def _synthesize(story, lang, prompt_path, voice_key):
    models = load_models()
    index = FREE_GPUS.get()  # waits until a GPU is free
    model = models[index]
    try:
        on_gpu = torch.cuda.is_available()
        with torch.cuda.device(index) if on_gpu else contextlib.nullcontext():
            cached = VOICE_CACHE.get((index, voice_key))
            if cached is None:
                print("Preparing voice", voice_key[:8], "on GPU", index)
                model.prepare_conditionals(prompt_path)
                VOICE_CACHE[(index, voice_key)] = model.conds
            else:
                print("Reusing voice", voice_key[:8], "on GPU", index)
                model.conds = cached
            # Seed only this GPU: torch.manual_seed would reset the other GPU mid-generation.
            if on_gpu:
                torch.cuda.manual_seed(42)
            else:
                torch.manual_seed(42)
            wav = model.generate(story, language_id=lang,cfg_weight=0.3, exaggeration=0.5)
        return wav.cpu(), model.sr
    finally:
        FREE_GPUS.put(index)
@app.get("/health")
def health():
    return {"ok": True, "cuda": torch.cuda.is_available(), "loaded": bool(MODELS), "gpus": len(MODELS), "voices_cached": len(VOICE_CACHE)}
@app.post("/generate")
async def generate(text: str = Form(...), language_id: str = Form("en"), audio: UploadFile = File(...), x_tts_secret: str | None = Header(default=None)):
    _check_secret(x_tts_secret)
    story = text.strip()
    if not story:
        raise HTTPException(400, "Text is required")
    prompt_bytes = await audio.read()
    if not prompt_bytes:
        raise HTTPException(400, "Audio file is required")
    voice_key = hashlib.sha256(prompt_bytes).hexdigest()
    suffix = Path(audio.filename or "prompt.wav").suffix or ".wav"
    prompt = Path(tempfile.gettempdir()) / ("suno_prompt_" + uuid.uuid4().hex + suffix)
    prompt.write_bytes(prompt_bytes)
    lang = _language_id(language_id)
    out = Path(tempfile.gettempdir()) / ("suno_out_" + uuid.uuid4().hex + ".wav")
    try:
        # Run in a thread: calling model.generate directly here blocks the server,
        # so a second request could never run on the other GPU at the same time.
        wav, sample_rate = await asyncio.to_thread(_synthesize, story, lang, str(prompt), voice_key)
        torchaudio.save(str(out), wav, sample_rate)
        return Response(content=out.read_bytes(), media_type="audio/wav")
    except Exception as error:
        raise HTTPException(500, f"Chatterbox failed: {error}") from error
    finally:
        prompt.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
