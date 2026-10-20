"""Kaggle GPU FastAPI: clone story speech with Chatterbox Multilingual, one process per GPU.

Paste into the Kaggle cell that starts with %%writefile /kaggle/working/chatterbox_server.py
and start it the same way as before (same uvicorn command, port and tunnel).

Two roles in one file:
- front (default): what uvicorn starts. Loads no model. Starts one worker process per GPU and
  sends each chunk to a free worker. Keeps the one public URL, so the backend is unchanged.
- worker (SUNO_ROLE=worker): one Chatterbox model on one GPU (CUDA_VISIBLE_DEVICES=<i>).

Why: with both models in one process, the two generation threads take turns on Python's GIL,
so each GPU ran at about half speed (~14 it/s instead of ~35 it/s on a T4).

A voice is prepared once (POST /voice) on one worker, and the same prepared voice (Conditionals
.pt bytes) is loaded on every worker. The backend downloads it (GET /voice/{key}/conds) and sends
it back after a Kaggle restart (POST /voice_conds), so the voice is not prepared again.
"""

from __future__ import annotations
import asyncio, contextlib, hashlib, io, os, subprocess, sys, tempfile, threading, uuid, wave
from pathlib import Path
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

SECRET = os.environ.get("COLAB_TTS_SECRET", "").strip()
ROLE = os.environ.get("SUNO_ROLE", "front").strip().lower()
WORKER_GPU = os.environ.get("WORKER_GPU", "0")
WORKER_BASE_PORT = int(os.environ.get("WORKER_BASE_PORT", "8100"))
CHATTERBOX_SRC = "/kaggle/working/chatterbox/src"
app = FastAPI(title="Suno Chatterbox Colab", version="2.0.0")


def _check_secret(x_tts_secret):
    if SECRET and (x_tts_secret or "").strip() != SECRET:
        raise HTTPException(401, "Invalid secret")


def _temp(prefix, suffix):
    return Path(tempfile.gettempdir()) / (prefix + uuid.uuid4().hex + suffix)


if ROLE == "worker":
    # ---------------- Worker: one model on one GPU ----------------
    if Path(CHATTERBOX_SRC).is_dir() and CHATTERBOX_SRC not in sys.path:
        sys.path.insert(0, CHATTERBOX_SRC)
    import torch, torchaudio

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # the only GPU this process sees
    MODEL = None
    GPU_LOCK = threading.Lock()  # one job at a time on this GPU
    VOICES = {}  # voice key -> Conditionals on this GPU
    CONDS_BYTES = {}  # voice key -> saved Conditionals (.pt bytes)

    def _language_id(value):
        text = (value or "en").strip().lower()
        return "hi" if text in {"hi", "hindi", "hi-in"} else "en"

    def _load_model():
        global MODEL
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        print("Loading ChatterboxMultilingualTTS on GPU", WORKER_GPU, flush=True)
        MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
        print("worker GPU", WORKER_GPU, "ready", flush=True)

    def _store_conds(voice_key, conds_bytes, label):
        from chatterbox.mtl_tts import Conditionals
        path = _temp("suno_conds_", ".pt")
        path.write_bytes(conds_bytes)
        try:
            VOICES[voice_key] = Conditionals.load(str(path), map_location=DEVICE)
        finally:
            path.unlink(missing_ok=True)
        CONDS_BYTES[voice_key] = conds_bytes
        print(label, "voice", voice_key[:8], "on GPU", WORKER_GPU, flush=True)

    def _prepare_voice(prompt_path, voice_key):
        with GPU_LOCK:
            if voice_key in CONDS_BYTES:
                return
            print("Preparing voice", voice_key[:8], "on GPU", WORKER_GPU, flush=True)
            MODEL.prepare_conditionals(prompt_path)
            path = _temp("suno_conds_", ".pt")
            try:
                MODEL.conds.save(str(path))
                conds_bytes = path.read_bytes()
            finally:
                path.unlink(missing_ok=True)
            # Load back from the saved bytes, so every GPU uses exactly the same voice.
            _store_conds(voice_key, conds_bytes, "Stored")

    def _synthesize(story, lang, voice_key):
        with GPU_LOCK:
            conds = VOICES.get(voice_key)
            if conds is None:
                raise KeyError(voice_key)
            MODEL.conds = conds
            if DEVICE.startswith("cuda"):
                torch.cuda.manual_seed(42)
            else:
                torch.manual_seed(42)
            wav = MODEL.generate(story, language_id=lang, cfg_weight=0.3, exaggeration=0.5)
            return wav.cpu(), MODEL.sr

    def _pcm16_wav(wav, sample_rate):
        # 16-bit mono WAV: half the size of torchaudio's default float WAV, so it crosses the
        # tunnel faster, and the backend's chunk cleanup (trim, fades, pause) can read it.
        samples = (wav.squeeze().clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).numpy()
        out = io.BytesIO()
        with wave.open(out, "wb") as dst:
            dst.setnchannels(1)
            dst.setsampwidth(2)
            dst.setframerate(int(sample_rate))
            dst.writeframes(samples.tobytes())
        return out.getvalue()

    @app.on_event("startup")
    async def _worker_startup():
        await asyncio.to_thread(_load_model)

    @app.get("/health")
    def worker_health():
        return {"ok": True, "cuda": torch.cuda.is_available(), "loaded": MODEL is not None, "gpu": WORKER_GPU, "voices_cached": len(CONDS_BYTES)}

    @app.post("/voice")
    async def worker_voice(audio: UploadFile = File(...)):
        prompt_bytes = await audio.read()
        if not prompt_bytes:
            raise HTTPException(400, "Audio file is required")
        voice_key = hashlib.sha256(prompt_bytes).hexdigest()
        if voice_key not in CONDS_BYTES:
            prompt = _temp("suno_prompt_", Path(audio.filename or "prompt.wav").suffix or ".wav")
            prompt.write_bytes(prompt_bytes)
            try:
                await asyncio.to_thread(_prepare_voice, str(prompt), voice_key)
            except Exception as error:
                raise HTTPException(500, f"Voice prepare failed: {error}") from error
            finally:
                prompt.unlink(missing_ok=True)
        return {"voice_key": voice_key}

    @app.get("/voice/{voice_key}/conds")
    def worker_conds_download(voice_key: str):
        conds_bytes = CONDS_BYTES.get(voice_key)
        if conds_bytes is None:
            raise HTTPException(404, "Voice not found")
        return Response(content=conds_bytes, media_type="application/octet-stream")

    @app.post("/voice_conds")
    async def worker_conds_upload(voice_key: str = Form(...), conds: UploadFile = File(...)):
        conds_bytes = await conds.read()
        if not conds_bytes:
            raise HTTPException(400, "Conds file is required")
        if voice_key not in CONDS_BYTES:
            try:
                await asyncio.to_thread(_store_conds, voice_key, conds_bytes, "Restored")
            except Exception as error:
                raise HTTPException(500, f"Voice restore failed: {error}") from error
        return {"voice_key": voice_key}

    @app.post("/generate")
    async def worker_generate(text: str = Form(...), language_id: str = Form("en"), voice_key: str = Form(...)):
        story = text.strip()
        if not story:
            raise HTTPException(400, "Text is required")
        if voice_key not in VOICES:
            raise HTTPException(404, "Voice not found")
        try:
            wav, sample_rate = await asyncio.to_thread(_synthesize, story, _language_id(language_id), voice_key)
            return Response(content=_pcm16_wav(wav, sample_rate), media_type="audio/wav")
        except KeyError:
            raise HTTPException(404, "Voice not found")
        except Exception as error:
            raise HTTPException(500, f"Chatterbox failed: {error}") from error

else:
    # ---------------- Front: the public server, one worker process per GPU ----------------
    import httpx

    WORKERS = []  # worker base URLs, one per GPU
    PROCS = []  # worker processes
    READY = []  # worker URLs that have loaded their model
    FAILED = []  # worker URLs that crashed twice while starting
    FREE = None  # asyncio.Queue of ready, idle worker URLs
    CLIENT = None
    CONDS_BYTES = {}  # voice key -> saved Conditionals, sent to every worker
    VOICE_LOCKS = {}  # voice key -> lock, so a voice is prepared only once
    GPU_COUNT = 0

    def _gpu_count():
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15).stdout
            return sum(1 for line in out.splitlines() if line.startswith("GPU "))
        except Exception:
            return 0

    def _spawn_worker(index, port):
        here = Path(__file__).resolve().parent
        paths = [CHATTERBOX_SRC] if Path(CHATTERBOX_SRC).is_dir() else []
        paths += [str(here), os.environ.get("PYTHONPATH", "")]
        env = dict(
            os.environ,
            SUNO_ROLE="worker",
            WORKER_GPU=str(index),
            CUDA_VISIBLE_DEVICES=str(index),
            COLAB_TTS_SECRET="",  # workers listen on 127.0.0.1 only; the front checks the secret
            PYTHONPATH=os.pathsep.join(p for p in paths if p),
        )
        command = [sys.executable, "-m", "uvicorn", f"{Path(__file__).stem}:app", "--host", "127.0.0.1", "--port", str(port)]
        return subprocess.Popen(command, env=env, cwd=str(here))

    async def _send_conds(url, voice_key):
        files = {"conds": ("voice.conds.pt", CONDS_BYTES[voice_key], "application/octet-stream")}
        response = await CLIENT.post(f"{url}/voice_conds", data={"voice_key": voice_key}, files=files)
        if response.status_code != 200:
            raise RuntimeError(f"worker {url} restore failed ({response.status_code}): {response.text[:200]}")

    async def _wait_ready(index, url, port):
        proc = _spawn_worker(index, port)
        PROCS.append(proc)
        restarted = False
        while True:
            if proc.poll() is not None:
                print("worker GPU", index, "exited with code", proc.returncode, flush=True)
                if restarted:
                    FAILED.append(url)
                    return
                # One retry covers a one-off startup race (for example a shared download folder).
                print("front: restarting worker GPU", index, flush=True)
                proc = _spawn_worker(index, port)
                PROCS.append(proc)
                restarted = True
            with contextlib.suppress(Exception):
                response = await CLIENT.get(f"{url}/health", timeout=5)
                if response.status_code == 200 and response.json().get("loaded"):
                    break
            await asyncio.sleep(2)
        READY.append(url)
        # Voices that arrived while this worker was loading.
        for voice_key in list(CONDS_BYTES):
            with contextlib.suppress(Exception):
                await _send_conds(url, voice_key)
        FREE.put_nowait(url)
        print("front: worker GPU", index, "ready", flush=True)

    @app.on_event("startup")
    async def _front_startup():
        global CLIENT, FREE, GPU_COUNT
        CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        FREE = asyncio.Queue()
        GPU_COUNT = _gpu_count()
        for index in range(max(1, GPU_COUNT)):
            WORKERS.append(f"http://127.0.0.1:{WORKER_BASE_PORT + index}")
        # Don't block startup: requests wait for a free worker until one has loaded.
        asyncio.create_task(_start_workers())

    async def _start_workers():
        # One after another: workers loading at the same time raced on shared download
        # folders (FileExistsError on /root/.pkuseg). Later workers load from the cache.
        for index, url in enumerate(WORKERS):
            port = WORKER_BASE_PORT + index
            print("front: starting worker GPU", index, "on port", port, flush=True)
            await _wait_ready(index, url, port)

    @app.on_event("shutdown")
    async def _front_shutdown():
        for proc in PROCS:
            with contextlib.suppress(Exception):
                proc.terminate()
        if CLIENT is not None:
            await CLIENT.aclose()

    async def _ensure_voice(prompt_bytes, filename):
        """Prepare a voice once on one worker, then load the same prepared voice on all workers."""
        voice_key = hashlib.sha256(prompt_bytes).hexdigest()
        lock = VOICE_LOCKS.setdefault(voice_key, asyncio.Lock())
        async with lock:
            if voice_key in CONDS_BYTES:
                return voice_key
            url = await FREE.get()
            try:
                files = {"audio": (filename or "prompt.wav", prompt_bytes, "application/octet-stream")}
                response = await CLIENT.post(f"{url}/voice", files=files)
                if response.status_code != 200:
                    raise HTTPException(response.status_code, response.text[:300])
                response = await CLIENT.get(f"{url}/voice/{voice_key}/conds")
                if response.status_code != 200:
                    raise HTTPException(500, f"Voice conds missing on worker: {response.text[:200]}")
                CONDS_BYTES[voice_key] = response.content
            finally:
                FREE.put_nowait(url)
            others = [other for other in list(READY) if other != url]
            await asyncio.gather(*(_send_conds(other, voice_key) for other in others))
            return voice_key

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "cuda": GPU_COUNT > 0,
            "loaded": bool(WORKERS) and len(READY) == len(WORKERS),
            "gpus": len(READY),
            "workers": len(WORKERS),
            "failed": len(FAILED),
            "voices_cached": len(CONDS_BYTES),
        }

    @app.post("/voice")
    async def voice(audio: UploadFile = File(...), x_tts_secret: str | None = Header(default=None)):
        _check_secret(x_tts_secret)
        prompt_bytes = await audio.read()
        if not prompt_bytes:
            raise HTTPException(400, "Audio file is required")
        try:
            voice_key = await _ensure_voice(prompt_bytes, audio.filename)
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(500, f"Voice prepare failed: {error}") from error
        return {"voice_key": voice_key}

    @app.get("/voice/{voice_key}/conds")
    def voice_conds_download(voice_key: str, x_tts_secret: str | None = Header(default=None)):
        _check_secret(x_tts_secret)
        conds_bytes = CONDS_BYTES.get(voice_key)
        if conds_bytes is None:
            raise HTTPException(404, "Voice not found")
        return Response(content=conds_bytes, media_type="application/octet-stream")

    @app.post("/voice_conds")
    async def voice_conds_upload(voice_key: str = Form(...), conds: UploadFile = File(...), x_tts_secret: str | None = Header(default=None)):
        _check_secret(x_tts_secret)
        conds_bytes = await conds.read()
        if not conds_bytes:
            raise HTTPException(400, "Conds file is required")
        if voice_key not in CONDS_BYTES:
            CONDS_BYTES[voice_key] = conds_bytes
            try:
                await asyncio.gather(*(_send_conds(url, voice_key) for url in list(READY)))
            except Exception as error:
                CONDS_BYTES.pop(voice_key, None)
                raise HTTPException(500, f"Voice restore failed: {error}") from error
            print("Restored voice", voice_key[:8], "on", len(READY), "GPU(s)", flush=True)
        return {"voice_key": voice_key}

    @app.post("/generate")
    async def generate(text: str = Form(...), language_id: str = Form("en"), voice_key: str | None = Form(None), audio: UploadFile | None = File(None), x_tts_secret: str | None = Header(default=None)):
        _check_secret(x_tts_secret)
        if not text.strip():
            raise HTTPException(400, "Text is required")
        if audio is not None:
            # Older clients send the prompt with every chunk: prepare it once, then use its key.
            prompt_bytes = await audio.read()
            if not prompt_bytes:
                raise HTTPException(400, "Audio file is required")
            voice_key = await _ensure_voice(prompt_bytes, audio.filename)
        if not voice_key or voice_key not in CONDS_BYTES:
            raise HTTPException(404, "Voice not found")
        form = {"text": text, "language_id": language_id, "voice_key": voice_key}
        url = await FREE.get()  # waits until a worker is free
        try:
            response = await CLIENT.post(f"{url}/generate", data=form)
            if response.status_code == 404 and voice_key in CONDS_BYTES:
                # Worker restarted and lost the voice: send it again and retry once.
                await _send_conds(url, voice_key)
                response = await CLIENT.post(f"{url}/generate", data=form)
        except Exception as error:
            raise HTTPException(500, f"Worker failed: {error}") from error
        finally:
            FREE.put_nowait(url)
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/octet-stream"),
        )
