import base64
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db import get_db
from app.models.voice import Voice
from app.schemas.voice import VoiceJson, VoicePatch, VoiceResponse
from app.services import voice_store
from app.services.audio import delete_file, save_bytes, save_recording
from app.services.voice_prep import prepare_voice

router = APIRouter(prefix="/voices", tags=["voices"])


def device_header(x_device_id: str | None = Header(default=None)) -> str:
    # No login yet: each phone sends the random id it saved on first launch.
    value = (x_device_id or "").strip()
    if not value or len(value) > 64:
        raise HTTPException(400, "X-Device-Id header is required")
    return value


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


def _out(request: Request, voice: Voice) -> VoiceResponse:
    base = str(request.base_url).rstrip("/")
    audio_url = f"{base}/v1/voices/{voice.id}/audio"
    return VoiceResponse(
        id=voice.id,
        name=voice.name,
        languages=voice.languages or ["English"],
        status=voice.status,
        is_default=voice.is_default,
        avatar=voice.avatar,
        gpu_prepared=voice.gpu_prepared,
        created_at=voice.created_at,
        isDefault=voice.is_default,
        audio_url=audio_url,
        audioUri=audio_url,
    )


async def _persist(
    request: Request,
    background: BackgroundTasks,
    db: AsyncSession,
    device_id: str,
    name: str,
    languages: list[str],
    path: Path,
) -> VoiceResponse:
    try:
        voice = await voice_store.create_voice(db, device_id, path.stem, name, languages, path)
        await db.commit()
    except SQLAlchemyError as error:
        delete_file(str(path))
        raise HTTPException(500, "Could not save the voice") from error
    # Clean the prompt and prepare it on the GPU now, so the first story skips that work.
    # Added after the commit, so the background job finds the row.
    background.add_task(prepare_voice, voice.id)
    return _out(request, voice)


async def _check_name(db: AsyncSession, device_id: str, name: str) -> str:
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name is required")
    if await voice_store.name_taken(db, device_id, name):
        raise HTTPException(400, f'You already have a voice named "{name}"')
    return name


@router.get("", response_model=list[VoiceResponse])
async def list_voices(
    request: Request,
    device_id: str = Depends(device_header),
    db: AsyncSession = Depends(get_db),
):
    try:
        voices = await voice_store.list_voices(db, device_id)
    except SQLAlchemyError as error:
        raise HTTPException(500, "Could not load voices") from error
    # The app lists voices on every launch, so this shows each phone's id (needed for the voices.json import).
    print(f"[voices] device {device_id}: {len(voices)} voice(s)", flush=True)
    return [_out(request, voice) for voice in voices]


@router.post("", response_model=VoiceResponse)
async def create_voice(
    request: Request,
    background: BackgroundTasks,
    name: str = Form(...),
    languages: str = Form("English"),
    audio: UploadFile = File(...),
    device_id: str = Depends(device_header),
    db: AsyncSession = Depends(get_db),
):
    name = await _check_name(db, device_id, name)
    voice_id = f"v_{uuid.uuid4().hex[:10]}"
    path = await save_recording(voice_id, audio)
    return await _persist(request, background, db, device_id, name, _parse_languages(languages), path)


@router.post("/json", response_model=VoiceResponse)
async def create_voice_json(
    body: VoiceJson,
    request: Request,
    background: BackgroundTasks,
    device_id: str = Depends(device_header),
    db: AsyncSession = Depends(get_db),
):
    name = await _check_name(db, device_id, body.name)
    payload = body.audio_base64.strip()
    if payload.startswith("data:"):
        payload = payload.split(",", 1)[-1]
    try:
        data = base64.b64decode(payload)
    except Exception as error:
        raise HTTPException(400, "Invalid audio data") from error
    if not data:
        raise HTTPException(400, "Audio file is required")
    voice_id = f"v_{uuid.uuid4().hex[:10]}"
    path = save_bytes(voice_id, data, body.filename)
    return await _persist(request, background, db, device_id, name, _parse_languages(body.languages), path)


# No device header: the app's audio player loads this URL directly. The voice id is random.
@router.get("/{voice_id}/audio")
async def get_voice_audio(voice_id: str, db: AsyncSession = Depends(get_db)):
    try:
        voice = await voice_store.get_voice(db, voice_id)
    except SQLAlchemyError as error:
        raise HTTPException(500, "Could not load the voice") from error
    if not voice:
        raise HTTPException(404, "Voice not found")
    file = voice_store.abs_path(voice.file_path)
    if not file:
        raise HTTPException(404, "Recording not found")
    if not file.exists():
        raise HTTPException(404, "Recording file missing")
    media = "audio/mp4" if file.suffix.lower() in {".m4a", ".mp4"} else "audio/mpeg"
    return FileResponse(file, media_type=media, filename=file.name)


@router.patch("/{voice_id}", response_model=VoiceResponse)
async def update_voice(
    voice_id: str,
    body: VoicePatch,
    request: Request,
    device_id: str = Depends(device_header),
    db: AsyncSession = Depends(get_db),
):
    try:
        voice = await voice_store.get_device_voice(db, device_id, voice_id)
        if not voice:
            raise HTTPException(404, "Voice not found")
        if body.name is not None and body.name.strip():
            name = body.name.strip()
            if await voice_store.name_taken(db, device_id, name, exclude_id=voice_id):
                raise HTTPException(400, f'You already have a voice named "{name}"')
            voice.name = name
        if body.avatar is not None:
            voice.avatar = body.avatar
        if body.is_default is True:
            await voice_store.set_default(db, device_id, voice_id)
        await db.commit()
        await db.refresh(voice)
    except SQLAlchemyError as error:
        raise HTTPException(500, "Could not update the voice") from error
    return _out(request, voice)


@router.delete("/{voice_id}")
async def delete_voice(
    voice_id: str,
    device_id: str = Depends(device_header),
    db: AsyncSession = Depends(get_db),
):
    try:
        voice = await voice_store.get_device_voice(db, device_id, voice_id)
        if not voice:
            raise HTTPException(404, "Voice not found")
        file_path = voice_store.abs_path(voice.file_path)
        await voice_store.delete_voice(db, voice)
        await db.commit()
    except SQLAlchemyError as error:
        raise HTTPException(500, "Could not delete the voice") from error
    # Files go only after the row is gone, so a failed delete keeps a working voice.
    if file_path:
        delete_file(str(file_path))
        delete_file(str(file_path.with_name(f"{file_path.stem}.prompt.wav")))
        if file_path.suffix.lower() != ".wav":
            delete_file(str(file_path.with_suffix(".wav")))  # ffmpeg conversion made by ensure_wav
    return {"ok": True}
