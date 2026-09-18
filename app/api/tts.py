from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services.audio import find_tts_path, media_type_for
from app.services.tts import TtsUnavailable, synthesize

router = APIRouter(prefix="/tts", tags=["tts"])


class TtsRequest(BaseModel):
    text: str
    language: str = "English"
    voice_id: str | None = None


@router.post("")
async def create_tts(body: TtsRequest, request: Request):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Text is required")
    # #region agent log
    try:
        import json
        import time
        with open(r"c:\Suno_Project\debug-98a9d4.log", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "sessionId": "98a9d4",
                "hypothesisId": "E",
                "location": "api/tts.py:create_tts",
                "message": "tts request",
                "data": {
                    "voiceId": body.voice_id or "",
                    "language": body.language,
                    "textLen": len(text),
                },
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion
    try:
        tts_id, path, duration, cloned = await synthesize(text, body.language, body.voice_id)
    except TtsUnavailable as error:
        raise HTTPException(503, str(error)) from error
    base = str(request.base_url).rstrip("/")
    audio_url = f"{base}/v1/tts/{tts_id}"
    return {
        "id": tts_id,
        "audio_url": audio_url,
        "duration": duration,
        "cloned": cloned,
    }


@router.get("/{tts_id}")
def get_tts_audio(tts_id: str):
    path = find_tts_path(tts_id)
    if not path:
        raise HTTPException(404, "Audio not found")
    return FileResponse(path, media_type=media_type_for(path), filename=path.name)
