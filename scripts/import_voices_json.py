"""One-time copy of data/voices.json into the PostgreSQL "voices" table.

Usage (from suno-tts/, venv active):
    python scripts/import_voices_json.py --device-id <the phone's X-Device-Id>

Voices whose id is already in the table are skipped. After a successful run the file is
renamed to voices.json.bak.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.db import close_db, sessionLocal  # noqa: E402
from app.models.voice import Voice  # noqa: E402
from app.services.voice_store import rel_path  # noqa: E402
from app.store import DATA_DIR  # noqa: E402

VOICES_JSON = DATA_DIR / "voices.json"


async def main(device_id: str) -> None:
    if not VOICES_JSON.exists():
        print(f"Nothing to import: {VOICES_JSON} not found")
        return
    items = json.loads(VOICES_JSON.read_text(encoding="utf-8"))
    added = skipped = 0
    async with sessionLocal() as db:
        has_default = False
        for item in items:
            if await db.get(Voice, item["id"]):
                skipped += 1
                continue
            is_default = bool(item.get("is_default")) and not has_default
            has_default = has_default or is_default
            db.add(Voice(
                id=item["id"],
                device_id=device_id,
                name=item["name"],
                languages=item.get("languages") or ["English"],
                # A voice stuck mid-prep is finished by resume_pending on the next start.
                status=item.get("status") or "Ready",
                is_default=is_default,
                avatar=item.get("avatar"),
                file_path=rel_path(item["file_path"]),
                prompt_path=rel_path(item["prompt_path"]) if item.get("prompt_path") else None,
                voice_key=item.get("voice_key"),
                gpu_prepared=bool(item.get("gpu_prepared")),
            ))
            added += 1
        await db.commit()
    await close_db()
    VOICES_JSON.rename(VOICES_JSON.with_name("voices.json.bak"))
    print(f"Imported {added} voice(s), skipped {skipped}. Renamed voices.json to voices.json.bak")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", required=True)
    asyncio.run(main(parser.parse_args().device_id.strip()))
