"""Deletes expired refresh tokens for every user, once a day. Revoked tokens stay until they
expire: /auth/refresh needs them to spot a stolen token being reused."""

import asyncio
from datetime import datetime, timezone

from sqlalchemy import delete

from app.database.db import sessionLocal
from app.models.refresh_token import RefreshToken

CLEANUP_EVERY_S = 86400


async def clean_refresh_tokens() -> int:
    """One sweep. Returns how many rows were deleted."""
    async with sessionLocal() as session:
        result = await session.execute(
            delete(RefreshToken).where(RefreshToken.expires_at < datetime.now(timezone.utc))
        )
        await session.commit()
        return result.rowcount or 0


async def watch_refresh_tokens() -> None:
    while True:
        try:
            removed = await clean_refresh_tokens()
            print(f"[tokens] removed {removed} expired refresh tokens", flush=True)
        except Exception as error:
            # A database problem only skips this sweep.
            print(f"[tokens] cleanup failed: {error}", flush=True)
        await asyncio.sleep(CLEANUP_EVERY_S)
