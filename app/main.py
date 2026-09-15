from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.tts import router as tts_router
from app.api.voices import router as voices_router
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
def health():
    return {"ok": True}
