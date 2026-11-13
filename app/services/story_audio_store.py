from datetime import datetime

from sqlalchemy import case, func, select, update
from sqlalchemy.dialects.postgresql import insert

from app.database.db import sessionLocal
from app.models.story import Story
from app.models.story_audio import StoryAudio


async def _run(story_key: str, statement) -> None:
    # A database error must never stop the story from streaming.
    try:
        async with sessionLocal() as db:
            await db.execute(statement)
            await db.commit()
    except Exception as error:
        print(f"[story_audio] {story_key}: {error}")


async def start_audio(
    story_key: str, story_id: int | None, voice_id: str, language: str, device_id: str, chunk_count: int
) -> None:
    # NULL when the story is not in the table (deleted, or a mock story), so the insert never fails.
    known_story_id = select(Story.id).where(Story.id == story_id).scalar_subquery()
    statement = insert(StoryAudio).values(
        story_key=story_key,
        story_id=known_story_id,
        voice_id=voice_id,
        language=language,
        device_id=device_id,
        status="generating",
        chunk_count=chunk_count,
    )
    is_ready = StoryAudio.status == "ready"
    statement = statement.on_conflict_do_update(
        # Played again after leaving halfway: back to generating. A ready row keeps its status,
        # count and updated_at; any row gets the story_id it was saved without.
        index_elements=[StoryAudio.story_key],
        set_={
            "status": case((is_ready, StoryAudio.status), else_="generating"),
            "chunk_count": case((is_ready, StoryAudio.chunk_count), else_=chunk_count),
            "updated_at": case((is_ready, StoryAudio.updated_at), else_=func.now()),
            "story_id": func.coalesce(statement.excluded.story_id, StoryAudio.story_id),
        },
    )
    await _run(story_key, statement)


def _update(story_key: str, **values):
    return update(StoryAudio).where(StoryAudio.story_key == story_key).values(**values)


async def mark_partial(story_key: str, chunks: list[str]) -> None:
    await _run(story_key, _update(story_key, status="partial", chunks_done=len(chunks), chunks=chunks))


async def mark_ready(story_key: str, final_key: str, duration_ms: int, chunk_count: int) -> None:
    await _run(story_key, _update(
        story_key, status="ready", final_key=final_key, duration_ms=duration_ms,
        chunks_done=chunk_count, chunks=None, partial_key=None, partial_samples=None,
    ))


async def mark_failed(story_key: str) -> None:
    await _run(story_key, _update(story_key, status="failed"))


async def mark_missing(story_key: str) -> None:
    # Its bucket file is gone: no longer ready, so the next play generates it again.
    await _run(story_key, _update(story_key, status="failed", final_key=None))


async def get_final_key(story_key: str) -> str | None:
    # A database error falls back to the local file, like the writes above.
    try:
        async with sessionLocal() as db:
            return await db.scalar(select(StoryAudio.final_key).where(StoryAudio.story_key == story_key))
    except Exception as error:
        print(f"[story_audio] {story_key}: {error}")
        return None


async def get_ready_key(story_key: str) -> str | None:
    """final_key of a finished story, None while it is not ready (or the database is down)."""
    try:
        async with sessionLocal() as db:
            return await db.scalar(
                select(StoryAudio.final_key).where(StoryAudio.story_key == story_key, StoryAudio.status == "ready")
            )
    except Exception as error:
        print(f"[story_audio] {story_key}: {error}")
        return None


async def ready_rows(story_keys: list[str]) -> list[tuple[str, str | None, datetime]]:
    """(story_key, final_key, last played or updated) of the given stories that are ready."""
    if not story_keys:
        return []
    async with sessionLocal() as db:
        rows = await db.execute(
            select(
                StoryAudio.story_key,
                StoryAudio.final_key,
                func.coalesce(StoryAudio.last_played_at, StoryAudio.updated_at),
            ).where(StoryAudio.story_key.in_(story_keys), StoryAudio.status == "ready")
        )
        return [tuple(row) for row in rows.all()]


async def partial_rows(story_keys: list[str]) -> list[tuple[str, datetime]]:
    """(story_key, last played or updated) of the given stories left unfinished or failed. A failed
    story is packed like a partial one, so its folder does not stay on disk forever and a replay
    continues from its made chunks."""
    if not story_keys:
        return []
    async with sessionLocal() as db:
        rows = await db.execute(
            select(
                StoryAudio.story_key,
                func.coalesce(StoryAudio.last_played_at, StoryAudio.updated_at),
            ).where(StoryAudio.story_key.in_(story_keys), StoryAudio.status.in_(("partial", "failed")))
        )
        return [tuple(row) for row in rows.all()]


async def set_partial(story_key: str, key: str, samples: list[int]) -> None:
    # Raises: the local chunks are only deleted once this is saved.
    async with sessionLocal() as db:
        await db.execute(_update(story_key, partial_key=key, partial_samples=samples))
        await db.commit()


async def get_partial(story_key: str) -> tuple[str, list[int]] | None:
    """(bucket key, frames per chunk) of a packed partial story, None if not packed (or the
    database is down)."""
    try:
        async with sessionLocal() as db:
            row = (await db.execute(
                select(StoryAudio.partial_key, StoryAudio.partial_samples).where(StoryAudio.story_key == story_key)
            )).first()
    except Exception as error:
        print(f"[story_audio] {story_key}: {error}")
        return None
    if row is None or not row[0] or not row[1]:
        return None
    return row[0], list(row[1])


async def clear_partial(story_key: str) -> None:
    await _run(story_key, _update(story_key, partial_key=None, partial_samples=None))


async def mark_interrupted() -> None:
    # Called at startup: a restart stopped every stream, so nothing is generating any more. The
    # chunks made stay on disk and playing again continues from them.
    await _run("startup", update(StoryAudio).where(StoryAudio.status == "generating").values(status="partial"))


async def save_position(story_key: str, position_ms: int, chunk_index: int) -> None:
    await _run(story_key, _update(
        story_key, position_ms=position_ms, last_chunk_index=chunk_index, last_played_at=func.now(),
    ))
