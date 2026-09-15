from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services.tts import synthesize

router = APIRouter(prefix="/tts", tags=["tts"])


class TtsRequest(BaseModel):
    text: str
    language: str = "English"


@router.post("")
async def create_tts(body: TtsRequest, request: Request):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Text is required")
    tts_id, path, duration = await synthesize(text, body.language)
    base = str(request.base_url).rstrip("/")
    audio_url = f"{base}/v1/tts/{tts_id}"
    return {"id": tts_id, "audio_url": audio_url, "duration": duration}


@router.get("/{tts_id}")
def get_tts_audio(tts_id: str):
    from app.services.audio import tts_path

    path = tts_path(tts_id)
    if not path.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(path, media_type="audio/mpeg", filename=f"{tts_id}.mp3")
