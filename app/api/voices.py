import asyncio
import base64
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_dependencies import get_current_user
from app.database.db import get_db
from app.models.user import User
from app.models.voice import Voice
from app.schemas.voice import VoiceJson, VoicePatch, VoiceResponse
from app.services import storage, voice_files, voice_store
from app.services.audio import delete_file, media_type_for, save_bytes, save_recording
from app.services.voice_prep import prepare_voice

router = APIRouter(prefix="/voices", tags=["voices"])


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
    user: User,
    name: str,
    languages: list[str],
    path: Path,
) -> VoiceResponse:
    try:
        voice = await voice_store.create_voice(db, user.id, path.stem, name, languages, path)
        await db.commit()
    except SQLAlchemyError as error:
        delete_file(str(path))
        raise HTTPException(500, "Could not save the voice") from error
    # Clean the prompt and prepare it on the GPU now, so the first story skips that work.
    # Added after the commit, so the background job finds the row.
    background.add_task(prepare_voice, voice.id)
    return _out(request, voice)


async def _check_name(db: AsyncSession, user: User, name: str) -> str:
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name is required")
    if await voice_store.name_taken(db, user.id, name):
        raise HTTPException(400, f'You already have a voice named "{name}"')
    return name


@router.get("", response_model=list[VoiceResponse])
async def list_voices(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Only the logged-in user's voices.
    try:
        voices = await voice_store.list_voices(db, user.id)
    except SQLAlchemyError as error:
        raise HTTPException(500, "Could not load voices") from error
    return [_out(request, voice) for voice in voices]


@router.post("", response_model=VoiceResponse)
async def create_voice(
    request: Request,
    background: BackgroundTasks,
    name: str = Form(...),
    languages: str = Form("English"),
    audio: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    name = await _check_name(db, user, name)
    voice_id = f"v_{uuid.uuid4().hex[:10]}"
    path = await save_recording(voice_id, audio)
    return await _persist(request, background, db, user, name, _parse_languages(languages), path)


@router.post("/json", response_model=VoiceResponse)
async def create_voice_json(
    body: VoiceJson,
    request: Request,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    name = await _check_name(db, user, body.name)
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
    return await _persist(request, background, db, user, name, _parse_languages(body.languages), path)


# No login header: the app's audio player loads this URL directly. The voice id is random.
@router.get("/{voice_id}/audio")
async def get_voice_audio(voice_id: str, db: AsyncSession = Depends(get_db)):
    try:
        voice = await voice_store.get_voice(db, voice_id)
    except SQLAlchemyError as error:
        raise HTTPException(500, "Could not load the voice") from error
    if not voice:
        raise HTTPException(404, "Voice not found")
    # The original recording: what the phone sent, smaller than the prompt WAV.
    file = voice_store.abs_path(voice.file_path)
    if not file:
        raise HTTPException(404, "Recording not found")
    if not file.exists():
        # Lost from the disk: play the bucket copy.
        if storage.storage_configured():
            key = voice_files.bucket_key(voice.id, file)
            try:
                if await asyncio.to_thread(storage.exists, key):
                    return RedirectResponse(await asyncio.to_thread(storage.signed_url, key), status_code=302)
            except Exception as error:
                print(f"[voice] {voice.id}: bucket link failed: {error}", flush=True)
        raise HTTPException(404, "Recording file missing")
    return FileResponse(file, media_type=media_type_for(file), filename=file.name)


@router.patch("/{voice_id}", response_model=VoiceResponse)
async def update_voice(
    voice_id: str,
    body: VoicePatch,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        voice = await voice_store.get_user_voice(db, user.id, voice_id)
        if not voice:
            raise HTTPException(404, "Voice not found")
        if body.name is not None and body.name.strip():
            name = body.name.strip()
            if await voice_store.name_taken(db, user.id, name, exclude_id=voice_id):
                raise HTTPException(400, f'You already have a voice named "{name}"')
            voice.name = name
        if body.avatar is not None:
            voice.avatar = body.avatar
        if body.is_default is True:
            await voice_store.set_default(db, user.id, voice_id)
        await db.commit()
        await db.refresh(voice)
    except SQLAlchemyError as error:
        raise HTTPException(500, "Could not update the voice") from error
    return _out(request, voice)


@router.delete("/{voice_id}")
async def delete_voice(
    voice_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        voice = await voice_store.get_user_voice(db, user.id, voice_id)
        if not voice:
            raise HTTPException(404, "Voice not found")
        await voice_store.delete_voice(db, voice)
        await db.commit()
    except SQLAlchemyError as error:
        raise HTTPException(500, "Could not delete the voice") from error
    # Files go only after the row is gone, so a failed delete keeps a working voice: the
    # recording, prompt, conds cache and the bucket copies.
    try:
        await asyncio.to_thread(voice_files.remove, voice_id)
    except Exception as error:
        print(f"[voice] {voice_id}: file cleanup failed: {error}", flush=True)
    return {"ok": True}
