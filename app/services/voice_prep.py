"""Prepare a saved voice once, right after upload, so story playback only generates."""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import select

from app.database.db import sessionLocal
from app.models.voice import Voice
from app.services import voice_files
from app.services.audio import prepare_prompt_wav
from app.services.chatterbox_client import colab_configured, upload_voice_to_gpu
from app.services.voice_store import abs_path, rel_path


async def _update(voice_id: str, **fields) -> bool:
    # Runs outside a request, so it opens its own session. Re-fetched before writing:
    # the voice may have been edited or deleted while preparing.
    async with sessionLocal() as db:
        voice = await db.get(Voice, voice_id)
        if not voice:
            return False
        for key, value in fields.items():
            setattr(voice, key, value)
        await db.commit()
        return True


def _back_up(voice_id: str, source: Path, prompt: Path) -> None:
    # A bucket problem never fails the voice: it works from the disk copy.
    try:
        voice_files.back_up(voice_id, [source, prompt])
    except Exception as error:
        print(f"[voice] {voice_id}: bucket backup failed, kept local: {error}", flush=True)


async def prepare_voice(voice_id: str) -> None:
    async with sessionLocal() as db:
        voice = await db.get(Voice, voice_id)
        source = abs_path(voice.file_path) if voice else None
    if source is None:
        return
    try:
        prompt = await asyncio.to_thread(prepare_prompt_wav, source)
        if not await _update(voice_id, prompt_path=rel_path(prompt)):
            return
        print(f"[voice] prompt ready {voice_id}", flush=True)
        await asyncio.to_thread(_back_up, voice_id, source, prompt)
        if colab_configured():
            voice_key = await upload_voice_to_gpu(str(prompt))
            await _update(voice_id, voice_key=voice_key, gpu_prepared=True)
            print(f"[voice] gpu prepared {voice_id} {voice_key[:8]}", flush=True)
    except Exception as error:
        # Playback still prepares/uploads on demand, so the voice stays usable.
        print(f"[voice] prepare failed {voice_id}: {error}", flush=True)
        await _update(voice_id, gpu_prepared=False)
    await _update(voice_id, status="Ready")


async def resume_pending() -> None:
    """Finish voices left in "Preparing" when the server stopped mid-job."""
    try:
        async with sessionLocal() as db:
            result = await db.execute(select(Voice.id).where(Voice.status == "Preparing"))
            pending = list(result.scalars().all())
    except Exception as error:
        # Startup must not fail because the database is down.
        print(f"[voice] resume skipped: {error}", flush=True)
        return
    for voice_id in pending:
        await prepare_voice(voice_id)
