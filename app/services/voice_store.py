"""Voice rows in PostgreSQL, one device to many voices. Audio files stay under data/."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.voice import Voice
from app.store import DATA_DIR


def rel_path(path: Path | str) -> str:
    # Stored relative to data/, so moving the server folder keeps the rows valid.
    path = Path(path)
    try:
        return path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def abs_path(value: str | None) -> Path | None:
    if not value:
        return None
    # An absolute value (imported from an old voices.json) is kept as it is by the join.
    return DATA_DIR / value


async def list_voices(db: AsyncSession, device_id: str) -> list[Voice]:
    result = await db.execute(
        select(Voice).where(Voice.device_id == device_id).order_by(Voice.created_at)
    )
    return list(result.scalars().all())


async def get_voice(db: AsyncSession, voice_id: str) -> Voice | None:
    return await db.get(Voice, voice_id)


async def get_device_voice(db: AsyncSession, device_id: str, voice_id: str) -> Voice | None:
    voice = await db.get(Voice, voice_id)
    # Another device's voice is reported as missing, not as forbidden.
    return voice if voice and voice.device_id == device_id else None


async def name_taken(db: AsyncSession, device_id: str, name: str, exclude_id: str | None = None) -> bool:
    query = select(Voice.id).where(Voice.device_id == device_id, Voice.name == name)
    if exclude_id:
        query = query.where(Voice.id != exclude_id)
    return (await db.execute(query.limit(1))).first() is not None


async def create_voice(
    db: AsyncSession,
    device_id: str,
    voice_id: str,
    name: str,
    languages: list[str],
    path: Path,
) -> Voice:
    has_voices = (
        await db.execute(select(Voice.id).where(Voice.device_id == device_id).limit(1))
    ).first() is not None
    voice = Voice(
        id=voice_id,
        device_id=device_id,
        name=name,
        languages=languages,
        status="Preparing",
        is_default=not has_voices,
        avatar="👤",
        file_path=rel_path(path),
        gpu_prepared=False,
    )
    db.add(voice)
    await db.flush()
    await db.refresh(voice)
    return voice


async def set_default(db: AsyncSession, device_id: str, voice_id: str) -> None:
    await db.execute(
        update(Voice)
        .where(Voice.device_id == device_id)
        .values(is_default=Voice.id == voice_id)
    )


async def delete_voice(db: AsyncSession, voice: Voice) -> None:
    device_id, was_default = voice.device_id, voice.is_default
    await db.delete(voice)
    await db.flush()
    if was_default:
        # The oldest remaining voice becomes the default.
        remaining = await list_voices(db, device_id)
        if remaining:
            remaining[0].is_default = True
