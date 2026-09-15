import base64
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services.audio import delete_file, save_bytes, save_recording
from app.store import load_voices, save_voices

router = APIRouter(prefix="/voices", tags=["voices"])


class VoicePatch(BaseModel):
    name: str | None = None
    is_default: bool | None = None
    avatar: str | None = None


class VoiceJson(BaseModel):
    name: str
    languages: str | list[str] = "English"
    audio_base64: str
    filename: str = "recording.m4a"


def _parse_languages(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()] or ["English"]
    value = (raw or "").strip()
    if not value:
        return ["English"]
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [part.strip() for part in value.split(",") if part.strip()]


def _with_urls(request: Request, voice: dict[str, Any]) -> dict[str, Any]:
    base = str(request.base_url).rstrip("/")
    return {
        **voice,
        "audio_url": f"{base}/v1/voices/{voice['id']}/audio",
        "audioUri": f"{base}/v1/voices/{voice['id']}/audio",
        "isDefault": voice.get("is_default", False),
        "status": voice.get("status", "Ready"),
    }


def _persist(request: Request, name: str, languages: list[str], path: Path) -> dict[str, Any]:
    voices = load_voices()
    voice = {
        "id": path.stem,
        "name": name.strip(),
        "languages": languages,
        "status": "Ready",
        "is_default": len(voices) == 0,
        "avatar": "👤",
        "file_path": str(path),
    }
    voices.append(voice)
    save_voices(voices)
    return _with_urls(request, voice)


@router.get("")
def list_voices(request: Request):
    return [_with_urls(request, voice) for voice in load_voices()]


@router.post("")
async def create_voice(
    request: Request,
    name: str = Form(...),
    languages: str = Form("English"),
    audio: UploadFile = File(...),
):
    if not name.strip():
        raise HTTPException(400, "Name is required")
    voice_id = f"v_{uuid.uuid4().hex[:10]}"
    path = await save_recording(voice_id, audio)
    return _persist(request, name, _parse_languages(languages), path)


@router.post("/json")
def create_voice_json(body: VoiceJson, request: Request):
    if not body.name.strip():
        raise HTTPException(400, "Name is required")
    payload = body.audio_base64.strip()
    if payload.startswith("data:"):
        print(payload,"😎😎")
        payload = payload.split(",", 1)[-1]
    try:
        data = base64.b64decode(payload)
    except Exception as error:
        raise HTTPException(400, "Invalid audio data") from error
    if not data:
        raise HTTPException(400, "Audio file is required")
    voice_id = f"v_{uuid.uuid4().hex[:10]}"
    path = save_bytes(voice_id, data, body.filename)
    return _persist(request, body.name, _parse_languages(body.languages), path)


@router.get("/{voice_id}/audio")
def get_voice_audio(voice_id: str):
    voice = next((item for item in load_voices() if item["id"] == voice_id), None)
    if not voice:
        raise HTTPException(404, "Voice not found")
    path = voice.get("file_path")
    if not path:
        raise HTTPException(404, "Recording not found")
    file = Path(path)
    if not file.exists():
        raise HTTPException(404, "Recording file missing")
    media = "audio/mp4" if file.suffix.lower() in {".m4a", ".mp4"} else "audio/mpeg"
    return FileResponse(file, media_type=media, filename=file.name)


@router.patch("/{voice_id}")
def update_voice(voice_id: str, body: VoicePatch, request: Request):
    voices = load_voices()
    voice = next((item for item in voices if item["id"] == voice_id), None)
    if not voice:
        raise HTTPException(404, "Voice not found")
    if body.name is not None:
        voice["name"] = body.name.strip() or voice["name"]
    if body.avatar is not None:
        voice["avatar"] = body.avatar
    if body.is_default is True:
        for item in voices:
            item["is_default"] = item["id"] == voice_id
    save_voices(voices)
    return _with_urls(request, voice)


@router.delete("/{voice_id}")
def delete_voice(voice_id: str):
    voices = load_voices()
    voice = next((item for item in voices if item["id"] == voice_id), None)
    if not voice:
        raise HTTPException(404, "Voice not found")
    delete_file(voice.get("file_path"))
    remaining = [item for item in voices if item["id"] != voice_id]
    if voice.get("is_default") and remaining:
        remaining[0]["is_default"] = True
    save_voices(remaining)
    return {"ok": True}
