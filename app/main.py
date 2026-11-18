import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.db import close_db
from app.api.tts import router as tts_router
from app.api.voices import router as voices_router
from app.routers.stories import router as story_router
from app.services.chatterbox_client import colab_configured
from app.services.gpu_status import gpu_status, watch_gpu
from app.services.story_audio_store import mark_interrupted
from app.services.story_cleanup import watch_story_files
from app.services.token_cleanup import watch_refresh_tokens
from app.services.voice_prep import resume_pending
from app.store import ensure_dirs

from slowapi.errors import RateLimitExceeded

from app.api.auth import router as auth_router
from app.core.rate_limit import limiter, rate_limit_exceeded

ensure_dirs()

app = FastAPI(title="Suno TTS", version="1.0.0", redirect_slashes=False)
# Limits login / signup attempts (see app/api/auth.py).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(voices_router, prefix="/v1")
app.include_router(tts_router, prefix="/v1")

 # story router is added here?
app.include_router(story_router, prefix="/api")
app.include_router(auth_router)


@app.on_event("startup")
async def resume_voice_prep():
    # Story streams were interrupted by the restart: their rows must not stay "generating".
    await mark_interrupted()
    # Voices still "Preparing" were interrupted by a restart; finish them in the background.
    asyncio.create_task(resume_pending())
    # Keeps the GPU state fresh in the background, so /health never waits for Kaggle.
    if colab_configured():
        _background_tasks.append(asyncio.create_task(watch_gpu()))
    # Removes finished stories from the server disk once they are safely in the bucket.
    _background_tasks.append(asyncio.create_task(watch_story_files()))
    # Deletes expired refresh tokens for all users, once a day.
    _background_tasks.append(asyncio.create_task(watch_refresh_tokens()))

@app.on_event("shutdown")
async def shutdown_event():
    await close_db()
    print("🔴 Shutting down... database  Connection closed ...")
    for task in _background_tasks:
        task.cancel()
    await asyncio.gather(*_background_tasks, return_exceptions=True)

# Strong references, so background loops are not garbage-collected.
_background_tasks: list[asyncio.Task[None]] = []


@app.get("/ping")
def ping():
    # Backend check only: answers at once and never touches Kaggle.
    return {"ok": True}


@app.get("/health")
def health():
    configured = colab_configured()
    gpu = gpu_status() if configured else {"reachable": False, "cuda": False, "loaded": False}
    return {
        "ok": True,
        "colab": configured,
        "colabConfigured": configured,
        "colabReachable": bool(gpu.get("reachable")),
        "colabCuda": bool(gpu.get("cuda")),
        "colabLoaded": bool(gpu.get("loaded")),
    }
