from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services.audio import find_tts_path, media_type_for
from app.services.story_stream import (
    cancel_story_play,
    cancel_story_stream,
    pause_story_stream,
    prefetch_story_start,
    resume_story_stream,
    start_story_stream,
    story_stream_chunk_exists,
    story_stream_chunk_path,
    story_stream_error,
    story_stream_final_path,
    wait_for_story_file,
)
from app.services.tts import TtsUnavailable, resolve_prompt, synthesize

router = APIRouter(prefix="/tts", tags=["tts"])


class TtsRequest(BaseModel):
    text: str
    language: str = "English"
    voice_id: str | None = None


class StoryStreamRequest(TtsRequest):
    voice_version: str | None = None
    # One tap on a story in the app; DELETE /stream/play/{play_id} stops listening to it.
    play_id: str | None = None


# Optional: when voice_id is missing or deleted, the device's default voice narrates.
DeviceId = Header(default=None, alias="X-Device-Id")


@router.post("")
async def create_tts(body: TtsRequest, request: Request, device_id: str | None = DeviceId):
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
        tts_id, path, duration, cloned = await synthesize(text, body.language, body.voice_id, device_id)
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


@router.post("/stream")
async def create_story_stream(body: StoryStreamRequest, request: Request, device_id: str | None = DeviceId):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Text is required")

    prompt = await resolve_prompt(body.voice_id, device_id)
    if prompt is None:
        raise HTTPException(503, "No prompt/voice recording available for streaming TTS")

    output = await start_story_stream(
        story_text=text,
        language=body.language,
        voice_id=prompt[0],
        voice_version=body.voice_version,
        prompt_wav_path=prompt[1],
        play_id=body.play_id,
    )

    base = str(request.base_url).rstrip("/")
    story_key = output.get("key")
    final_audio_url = f"{base}/v1/tts/stream/{story_key}/final"
    if output.get("cancelled"):
        return {
            "streaming": False,
            "cached": False,
            "cancelled": True,
            "story_key": story_key,
        }

    if output.get("cached"):
        return {
            "streaming": True,
            "cached": True,
            "story_key": story_key,
            "audio_url": final_audio_url,
        }

    if not output.get("chunk_count"):
        return {
            "streaming": False,
            "cached": False,
            "story_key": story_key,
            "chunk_count": 0,
        }

    return {
        "streaming": True,
        "cached": False,
        "story_key": story_key,
        "first_chunk_url": f"{base}/v1/tts/stream/{story_key}/chunk/0",
        "final_audio_url": final_audio_url,
        "chunk_count": output.get("chunk_count", 0),
        "chunk_words": output.get("chunk_words", []),
    }


@router.post("/stream/prefetch")
async def prefetch_story_stream(body: StoryStreamRequest, device_id: str | None = DeviceId):
    # The app calls this for the next story while the current one plays: only its first chunks are
    # queued, and a free GPU makes them between the current story's chunks.
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Text is required")

    prompt = await resolve_prompt(body.voice_id, device_id)
    if prompt is None:
        raise HTTPException(503, "No prompt/voice recording available for streaming TTS")

    story_key = prefetch_story_start(
        story_text=text,
        language=body.language,
        voice_id=prompt[0],
        voice_version=body.voice_version,
        prompt_wav_path=prompt[1],
    )
    return {"story_key": story_key}


@router.delete("/stream/{story_key}")
def delete_story_stream(story_key: str):
    # The app calls this when the listener leaves a story that is still generating.
    return {"cancelled": cancel_story_stream(story_key)}


@router.delete("/stream/play/{play_id}")
def delete_story_play(play_id: str):
    # The app calls this when the listener leaves a play, even one still waiting for its first
    # part. The story keeps generating while another play (or device) still listens to it.
    return {"cancelled": cancel_story_play(play_id)}


@router.post("/stream/{story_key}/pause")
def pause_story_stream_route(story_key: str, chunk: int = 0):
    # The listener paused on `chunk`: only a few chunks past it keep generating.
    return {"paused": pause_story_stream(story_key, chunk)}


@router.post("/stream/{story_key}/resume")
def resume_story_stream_route(story_key: str):
    return {"resumed": resume_story_stream(story_key)}


STREAM_WAIT_MAX_S = 25.0


async def _wait_response(story_key: str, path: Path, timeout: float) -> dict[str, str | None]:
    status = await wait_for_story_file(story_key, path, max(1.0, min(timeout, STREAM_WAIT_MAX_S)))
    return {"status": status, "detail": story_stream_error(story_key) if status == "failed" else None}


# Long-poll readiness checks: the request is held until the file exists, generation fails,
# or `timeout` seconds pass, so the app needs about one request per chunk instead of polling.
@router.get("/stream/{story_key}/chunk/{chunk_index}/wait")
async def wait_story_stream_chunk(story_key: str, chunk_index: int, timeout: float = STREAM_WAIT_MAX_S):
    return await _wait_response(story_key, story_stream_chunk_path(story_key, chunk_index), timeout)


@router.get("/stream/{story_key}/final/wait")
async def wait_story_stream_final(story_key: str, timeout: float = STREAM_WAIT_MAX_S):
    return await _wait_response(story_key, story_stream_final_path(story_key), timeout)


@router.get("/stream/{story_key}/chunk/{chunk_index}")
def get_story_stream_chunk(story_key: str, chunk_index: int):
    if not story_stream_chunk_exists(story_key, chunk_index):
        error = story_stream_error(story_key)
        if error:
            raise HTTPException(500, f"Story stream failed: {error}")
        raise HTTPException(404, "Story stream chunk not found")
    path = story_stream_chunk_path(story_key, chunk_index)
    return FileResponse(path, media_type=media_type_for(path), filename=path.name)


@router.get("/stream/{story_key}/final")
def get_story_stream_final(story_key: str):
    path = story_stream_final_path(story_key)
    if not path.exists() or path.stat().st_size == 0:
        error = story_stream_error(story_key)
        if error:
            raise HTTPException(500, f"Story stream failed: {error}")
        raise HTTPException(404, "Story stream final audio not found")
    return FileResponse(path, media_type=media_type_for(path), filename=path.name)


@router.get("/{tts_id}")
def get_tts_audio(tts_id: str):
    path = find_tts_path(tts_id)
    if not path:
        raise HTTPException(404, "Audio not found")
    return FileResponse(path, media_type=media_type_for(path), filename=path.name)
