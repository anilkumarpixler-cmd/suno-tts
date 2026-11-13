"""One-time move of data/recordings/ to the new voice file layout (app/services/voice_files.py).

For every voice in the table:
  - file_path points at the original recording (<id>.m4a) instead of the converted <id>.wav,
    which is then deleted (the prompt is made straight from the original now);
  - the original and the prompt are copied to the bucket under voices/<id>/.
Files of voices no longer in the table (left by older deletes) are removed.

Usage (from suno-tts/, venv active):
    python scripts/migrate_voice_files.py            # shows what it would do
    python scripts/migrate_voice_files.py --apply    # does it
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.database.db import close_db, sessionLocal  # noqa: E402
from app.models.voice import Voice  # noqa: E402
from app.services import storage, voice_files  # noqa: E402
from app.services.voice_store import abs_path, rel_path  # noqa: E402
from app.store import RECORDINGS_DIR  # noqa: E402


def _original(voice_id: str) -> Path | None:
    # <id>.<ext> other than the converted .wav, the prompt and the conds cache.
    for path in sorted(RECORDINGS_DIR.glob(f"{voice_id}.*")):
        if path.suffix.lower() not in {".wav", ".pt"} and path.name.count(".") == 1:
            return path
    return None


async def main(apply: bool) -> None:
    act = "" if apply else "[dry run] "
    async with sessionLocal() as db:
        voices = list((await db.execute(select(Voice))).scalars().all())
        for voice in voices:
            source = abs_path(voice.file_path)
            prompt = abs_path(voice.prompt_path)
            original = _original(voice.id)
            if source and source.suffix.lower() == ".wav" and original:
                print(f"{act}{voice.id} ({voice.name}): file_path -> {original.name}, delete {source.name}")
                if apply:
                    # The prompt must not look older than the original, or it is rebuilt and the
                    # GPU's voice_key changes.
                    if prompt and prompt.exists() and prompt.stat().st_mtime < original.stat().st_mtime:
                        os.utime(prompt)
                    voice.file_path = rel_path(original)
                    await db.commit()
                    if prompt is None or prompt != source:
                        source.unlink(missing_ok=True)
                source = original
            if storage.storage_configured():
                files = [path for path in (source, prompt) if path and path.exists()]
                print(f"{act}{voice.id}: back up {', '.join(path.name for path in files)}")
                if apply:
                    await asyncio.to_thread(voice_files.back_up, voice.id, files)

        known = {voice.id for voice in voices}
        for path in sorted(RECORDINGS_DIR.iterdir()):
            if path.is_file() and path.name.split(".", 1)[0] not in known:
                print(f"{act}orphan: delete {path.name}")
                if apply:
                    path.unlink()
    await close_db()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="make the changes (default: only show them)")
    asyncio.run(main(parser.parse_args().apply))
