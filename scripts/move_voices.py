"""Move a phone's voices (and its saved story audio) from an old X-Device-Id to a new one.

Needed after the app was reinstalled before it kept its id: the voices are still in the table,
just under the old id.

Usage (from suno-tts/, venv active):
    python scripts/move_voices.py --list                       # every device id and its voices
    python scripts/move_voices.py --from <old id> --to <new id>
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import func, select, update

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.db import close_db, sessionLocal  # noqa: E402
from app.models.story_audio import StoryAudio  # noqa: E402
from app.models.voice import Voice  # noqa: E402


async def list_devices() -> None:
    async with sessionLocal() as db:
        rows = (await db.execute(
            select(Voice.device_id, func.count(), func.max(Voice.created_at), func.array_agg(Voice.name))
            .group_by(Voice.device_id)
            .order_by(func.max(Voice.created_at).desc())
        )).all()
    if not rows:
        print("No voices in the table")
    for device_id, count, newest, names in rows:
        print(f"{device_id}  {count} voice(s), newest {newest:%Y-%m-%d %H:%M}  {', '.join(names)}")


async def move(old: str, new: str) -> None:
    async with sessionLocal() as db:
        moving = (await db.scalars(select(Voice).where(Voice.device_id == old))).all()
        if not moving:
            print(f"No voices under {old}")
            return
        target = (await db.scalars(select(Voice).where(Voice.device_id == new))).all()
        taken = {v.name for v in target}
        has_default = any(v.is_default for v in target)
        renamed = 0
        for voice in moving:
            # Names are unique per device: keep both voices by marking the moved one.
            if voice.name in taken:
                base, n = f"{voice.name} (old)", 2
                name = base
                while name in taken:
                    name, n = f"{base} {n}", n + 1
                voice.name = name[:100]
                renamed += 1
            taken.add(voice.name)
            # One default per device: the new phone's default wins.
            if has_default:
                voice.is_default = False
            has_default = has_default or voice.is_default
            voice.device_id = new
        stories = await db.execute(
            update(StoryAudio).where(StoryAudio.device_id == old).values(device_id=new)
        )
        await db.commit()
    print(f"Moved {len(moving)} voice(s) ({renamed} renamed) and {stories.rowcount} story audio(s) "
          f"from {old} to {new}")


async def main(args: argparse.Namespace) -> None:
    try:
        if args.list:
            await list_devices()
        else:
            await move(args.old.strip(), args.new.strip())
    finally:
        await close_db()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="show every device id and its voices")
    parser.add_argument("--from", dest="old", help="the old X-Device-Id")
    parser.add_argument("--to", dest="new", help="the new X-Device-Id")
    parsed = parser.parse_args()
    if not parsed.list and not (parsed.old and parsed.new):
        parser.error("use --list, or both --from and --to")
    if not parsed.list and len(parsed.new.strip()) > 64:
        parser.error("--to is longer than 64 characters")
    asyncio.run(main(parsed))
