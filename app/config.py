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
