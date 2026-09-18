from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.tts import router as tts_router
from app.api.voices import router as voices_router
from app.services.chatterbox_client import colab_configured, ping_colab
from app.store import ensure_dirs

ensure_dirs()

app = FastAPI(title="Suno TTS", version="1.0.0", redirect_slashes=False)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(voices_router, prefix="/v1")
app.include_router(tts_router, prefix="/v1")


@app.get("/health")
async def health():
    configured = colab_configured()
    ping = (
        await ping_colab()
        if configured
        else {"reachable": False, "cuda": False, "loaded": False}
    )
    return {
        "ok": True,
        "colab": configured,
        "colabConfigured": configured,
        "colabReachable": bool(ping.get("reachable")),
        "colabCuda": bool(ping.get("cuda")),
        "colabLoaded": bool(ping.get("loaded")),
    }
