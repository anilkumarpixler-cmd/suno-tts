from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_dotenv()

COLAB_TTS_URL = os.environ.get("COLAB_TTS_URL", "").strip().rstrip("/")
COLAB_TTS_SECRET = os.environ.get("COLAB_TTS_SECRET", "").strip()
COLAB_TTS_TIMEOUT = float(os.environ.get("COLAB_TTS_TIMEOUT", "180"))

# S3-compatible bucket for finished stories (Backblaze B2 now, R2 later: only these values change).
STORAGE_ENDPOINT = os.environ.get("STORAGE_ENDPOINT", "").strip().rstrip("/")
STORAGE_REGION = os.environ.get("STORAGE_REGION", "").strip()
STORAGE_ACCESS_KEY_ID = os.environ.get("STORAGE_ACCESS_KEY_ID", "").strip()
STORAGE_SECRET_ACCESS_KEY = os.environ.get("STORAGE_SECRET_ACCESS_KEY", "").strip()
STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET", "").strip()
