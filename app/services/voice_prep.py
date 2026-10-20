"""Prepare a saved voice once, right after upload, so story playback only generates."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.services.audio import ensure_wav, prepare_prompt_wav
from app.services.chatterbox_client import colab_configured, upload_voice_to_gpu
from app.store import load_voices, save_voices


def _update(voice_id: str, **fields) -> bool:
    # Re-load before writing: the voice may have been edited or deleted while preparing.
    voices = load_voices()
    voice = next((item for item in voices if item.get("id") == voice_id), None)
    if not voice:
        return False
    voice.update(fields)
    save_voices(voices)
    return True


def _clean_prompt(source: Path) -> Path:
    prompt = ensure_wav(source)
    if prompt.suffix.lower() == ".wav":
        prompt = prepare_prompt_wav(prompt)
    return prompt


async def prepare_voice(voice_id: str) -> None:
    voice = next((item for item in load_voices() if item.get("id") == voice_id), None)
    if not voice or not voice.get("file_path"):
        return
    source = Path(voice["file_path"])
    try:
        prompt = await asyncio.to_thread(_clean_prompt, source)
        if not _update(voice_id, prompt_path=str(prompt)):
            return
        print(f"[voice] prompt ready {voice_id}", flush=True)
        if colab_configured():
            voice_key = await upload_voice_to_gpu(str(prompt))
            _update(voice_id, voice_key=voice_key, gpu_prepared=True)
            print(f"[voice] gpu prepared {voice_id} {voice_key[:8]}", flush=True)
    except Exception as error:
        # Playback still prepares/uploads on demand, so the voice stays usable.
        print(f"[voice] prepare failed {voice_id}: {error}", flush=True)
        _update(voice_id, gpu_prepared=False)
    _update(voice_id, status="Ready")


async def resume_pending() -> None:
    """Finish voices left in "Preparing" when the server stopped mid-job."""
    for voice in load_voices():
        if voice.get("status") == "Preparing":
            await prepare_voice(voice["id"])
