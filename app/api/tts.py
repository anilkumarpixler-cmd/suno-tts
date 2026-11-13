import asyncio
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel
from app.services import hls, storage
from app.services.story_audio_store import get_final_key, save_position

from app.services.audio import find_tts_path, media_type_for
from app.services.chatterbox_client import colab_configured
from app.services.gpu_status import gpu_status
from app.services.story_stream import (
    cancel_story_play,
    cancel_story_stream,
    keep_stream_alive,
    note_listening,
    pause_story_stream,
    prefetch_story_start,
    resume_story_stream,
    start_story_stream,
    story_cache_key,
    story_stream_chunk_exists,
    story_stream_chunk_path,
    story_stream_dir,
    story_stream_error,
    story_stream_final_path,
    bucket_has_story,
    story_busy,
    story_ready,
    story_stream_stopped,
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
    story_id: int | None = None       # ← add
    # One tap on a story in the app; DELETE /stream/play/{play_id} stops listening to it.
    play_id: str | None = None
    # Saved position of a story heard halfway: generation continues from that chunk.
    resume_ms: int = 0

class ProgressRequest(BaseModel):
    position_ms: int
    chunk_index: int = 0



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

    # A saved story plays from storage without the GPU; a new one needs it, so answer at once
    # instead of making the app wait on a GPU that is down.
    story_key = story_cache_key(text, prompt[0], body.language, body.voice_version)
    if colab_configured() and not gpu_status().get("loaded") and not await story_ready(story_key, check_bucket=True):
        raise HTTPException(503, "Cloned voice offline. Playing in System Voice.")

    output = await start_story_stream(
        story_text=text,
        language=body.language,
        voice_id=prompt[0],
        voice_version=body.voice_version,
        prompt_wav_path=prompt[1],
        play_id=body.play_id,
        story_id=body.story_id,       # ← add
        device_id=device_id,          # ← add
        resume_ms=max(0, body.resume_ms),
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
        # HLS: one URL for the whole story while it is generated (gapless at chunk changes).
        "playlist_url": f"{base}/v1/tts/stream/{story_key}/playlist.m3u8",
        "status_url": f"{base}/v1/tts/stream/{story_key}/status",
        "start_chunk": output.get("start_chunk", 0),
        "start_ms": output.get("start_ms", 0),
        "resume_reset": output.get("resume_reset", False),
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

    story_key = await prefetch_story_start(
        story_text=text,
        language=body.language,
        voice_id=prompt[0],
        voice_version=body.voice_version,
        prompt_wav_path=prompt[1],
        device_id=device_id,
    )
    return {"story_key": story_key}


# The stream routes below are async so they run on the event loop that owns the stream state.
@router.delete("/stream/{story_key}")
async def delete_story_stream(story_key: str):
    # The app calls this when the listener leaves a story that is still generating.
    return {"cancelled": cancel_story_stream(story_key)}


@router.delete("/stream/play/{play_id}")
async def delete_story_play(play_id: str):
    # The app calls this when the listener leaves a play, even one still waiting for its first
    # part. The story keeps generating while another play (or device) still listens to it.
    return {"cancelled": cancel_story_play(play_id)}


@router.post("/stream/{story_key}/pause")
async def pause_story_stream_route(story_key: str, chunk: int = 0, position_ms: int | None = None):
    # The listener paused on `chunk`: only a few chunks past it keep generating. The HLS app only
    # knows its time, so it sends `position_ms` and the chunk is found here.
    if position_ms is not None:
        folder = story_stream_dir(story_key)
        found = hls.chunk_at(folder, max(0, position_ms))
        chunk = found[0] if found is not None else hls.made_count(folder)
    return {"paused": pause_story_stream(story_key, chunk)}


@router.post("/stream/{story_key}/resume")
async def resume_story_stream_route(story_key: str):
    # Back after a long pause: a stream stopped meanwhile continues from the chunks on disk.
    await keep_stream_alive(story_key)
    return {"resumed": resume_story_stream(story_key)}


STREAM_WAIT_MAX_S = 25.0


async def _wait_response(story_key: str, path: Path, timeout: float) -> dict[str, str | None]:
    status = await wait_for_story_file(story_key, path, max(1.0, min(timeout, STREAM_WAIT_MAX_S)))
    return {"status": status, "detail": story_stream_error(story_key) if status == "failed" else None}


# Long-poll readiness checks: the request is held until the file exists, generation fails,
# or `timeout` seconds pass, so the app needs about one request per chunk instead of polling.
@router.get("/stream/{story_key}/chunk/{chunk_index}/wait")
async def wait_story_stream_chunk(story_key: str, chunk_index: int, timeout: float = STREAM_WAIT_MAX_S):
    # Cleaned up after the story was saved: the app plays the saved file from here instead.
    if not story_stream_dir(story_key).exists() and await story_ready(story_key):
        return {"status": "final", "detail": None}
    # The app is still open on this story: it is not stopped for nobody listening.
    await keep_stream_alive(story_key)
    result = await _wait_response(story_key, story_stream_chunk_path(story_key, chunk_index), timeout)
    if result["status"] == "pending" and story_stream_stopped(story_key, chunk_index):
        # Nothing will make this chunk (e.g. the server restarted): tell the app instead of waiting forever.
        return {"status": "failed", "detail": "This story stopped. Play it again to continue."}
    return result


@router.get("/stream/{story_key}/final/wait")
async def wait_story_stream_final(story_key: str, timeout: float = STREAM_WAIT_MAX_S):
    # Also ready once the local files were cleaned up and only the bucket copy is left.
    if await story_ready(story_key):
        return {"status": "ready", "detail": None}
    await keep_stream_alive(story_key)
    return await _wait_response(story_key, story_stream_final_path(story_key), timeout)


@router.get("/stream/{story_key}/chunk/{chunk_index}")
async def get_story_stream_chunk(story_key: str, chunk_index: int):
    if not story_stream_chunk_exists(story_key, chunk_index):
        error = story_stream_error(story_key)
        if error:
            raise HTTPException(500, f"Story stream failed: {error}")
        raise HTTPException(404, "Story stream chunk not found")
    # The app fetches a chunk when it starts playing it: the server stays a few chunks ahead.
    note_listening(story_key, chunk_index)
    path = story_stream_chunk_path(story_key, chunk_index)
    return FileResponse(path, media_type=media_type_for(path), filename=path.name)


@router.get("/stream/{story_key}/final")
async def get_story_stream_final(story_key: str):
    # Uploaded: send the player to a fresh signed link. The app keeps this backend URL, so a
    # saved story never holds a link that expires.
    final_key = await get_final_key(story_key)
    # A signed link to a file the bucket lost would only fail in the player: the local file plays, if any.
    if storage.is_stored(final_key) and storage.storage_configured() and await bucket_has_story(story_key, final_key):
        try:
            url = await asyncio.to_thread(storage.signed_url, final_key)
            return RedirectResponse(url, status_code=302)
        except Exception as error:
            print(f"[storage] {story_key[:8]}: signed link failed, serving local: {error}", flush=True)
    path = story_stream_final_path(story_key)
    if not path.exists() or path.stat().st_size == 0:
        error = story_stream_error(story_key)
        if error:
            raise HTTPException(500, f"Story stream failed: {error}")
        raise HTTPException(404, "Story audio missing. Play it again.")
    return FileResponse(path, media_type=media_type_for(path), filename=path.name)


SEGMENT_CACHE = "public, max-age=31536000, immutable"


async def _story_ended(story_key: str) -> bool:
    # Ready or failed: nothing more is added, so the playlist is closed.
    return await story_ready(story_key) or story_stream_error(story_key) is not None


@router.get("/stream/{story_key}/playlist.m3u8")
async def get_story_playlist(story_key: str):
    folder = story_stream_dir(story_key)
    if not folder.is_dir():
        # Cleaned up (saved in the bucket) or never started: the app plays /final or starts again.
        raise HTTPException(404, "Story stream not found")
    # The player reloads this while the story plays: it counts as listening.
    await keep_stream_alive(story_key)
    playlist = hls.build_playlist(folder, await _story_ended(story_key))
    return Response(playlist, media_type=hls.PLAYLIST_TYPE, headers={"Cache-Control": "no-cache"})


async def _segment_file(story_key: str, index: int) -> Path:
    folder = story_stream_dir(story_key)
    if index < 0 or index >= hls.made_count(folder):
        raise HTTPException(404, "Segment not made yet")
    try:
        return await hls.ensure_segment(folder, index)
    except FileNotFoundError as error:
        raise HTTPException(404, "Segment not made yet") from error
    except Exception as error:
        raise HTTPException(500, f"Segment failed: {error}") from error


@router.get("/stream/{story_key}/init.mp4")
async def get_story_init(story_key: str):
    folder = story_stream_dir(story_key)
    if not hls.init_path(folder).exists():
        # Written with the first segment.
        await _segment_file(story_key, 0)
    return FileResponse(hls.init_path(folder), media_type=hls.SEGMENT_TYPE, headers={"Cache-Control": SEGMENT_CACHE})


@router.get("/stream/{story_key}/seg/{index}.m4s")
async def get_story_segment(story_key: str, index: int):
    path = await _segment_file(story_key, index)
    # The player fetches a few segments ahead of what it plays: the server stays a few chunks ahead.
    note_listening(story_key, index)
    return FileResponse(path, media_type=hls.SEGMENT_TYPE, headers={"Cache-Control": SEGMENT_CACHE})


@router.get("/stream/{story_key}/status")
async def get_story_status(story_key: str):
    folder = story_stream_dir(story_key)
    error = story_stream_error(story_key)
    if await story_ready(story_key):
        status = "ready"
    elif error:
        status = "failed"
    elif story_busy(story_key):
        status = "generating"
    else:
        # Left and not generating (skipped, idle, or the server restarted): POST /stream continues it.
        status = "stopped"
    return {
        "status": status,
        "made_chunks": hls.made_count(folder),
        "made_ms": hls.made_ms(folder),
        "detail": error if status == "failed" else None,
    }


@router.get("/{tts_id}")
def get_tts_audio(tts_id: str):
    path = find_tts_path(tts_id)
    if not path:
        raise HTTPException(404, "Audio not found")
    return FileResponse(path, media_type=media_type_for(path), filename=path.name)

@router.post("/stream/{story_key}/progress")
async def save_story_progress(story_key: str, body: ProgressRequest):
    # The app calls this every 15–30 seconds and on pause.
    await save_position(story_key, body.position_ms, body.chunk_index)
    return {"saved": True}