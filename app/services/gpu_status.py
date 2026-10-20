from __future__ import annotations

import asyncio
import time

# How often the background check asks Kaggle's /health while it answers. It never generates audio.
GPU_CHECK_INTERVAL_S = 20.0
# Circuit breaker: after this many failed checks in a row, wait longer between checks, so a dead
# or restarting Kaggle is not asked every 20 s. The wait doubles per failure up to the maximum.
GPU_FAILS_BEFORE_BACKOFF = 3
GPU_CHECK_MAX_INTERVAL_S = 120.0

# Last known GPU state, so /health answers at once instead of waiting for Kaggle.
_status: dict = {"reachable": False, "cuda": False, "loaded": False, "error": "not checked yet", "checked_at": 0.0}
# Failed checks (or failed chunks) in a row; reset by the first success.
_fails = 0


def gpu_status() -> dict:
    """The saved GPU state. No network call."""
    return dict(_status)


def mark_gpu(up: bool, error: str | None = None) -> None:
    """A real chunk worked (up) or Kaggle could not be reached (down): no need to wait for the next check."""
    global _fails
    if up:
        _fails = 0
        _status.update(reachable=True, cuda=True, loaded=True, error=None, checked_at=time.time())
    else:
        _fails += 1
        _status.update(reachable=False, loaded=False, error=error or "generate failed", checked_at=time.time())


def next_check_delay() -> float:
    """20 s while the GPU answers; after GPU_FAILS_BEFORE_BACKOFF failures, 40 s, 80 s, then 120 s."""
    if _fails < GPU_FAILS_BEFORE_BACKOFF:
        return GPU_CHECK_INTERVAL_S
    doublings = _fails - GPU_FAILS_BEFORE_BACKOFF + 1
    return min(GPU_CHECK_INTERVAL_S * 2**doublings, GPU_CHECK_MAX_INTERVAL_S)


async def check_gpu() -> dict:
    # Imported here: chatterbox_client imports this module to call mark_gpu.
    from app.services.chatterbox_client import ping_colab

    ping = await ping_colab()
    if ping.get("reachable") and ping.get("loaded"):
        mark_gpu(True)
        _status["cuda"] = bool(ping.get("cuda"))
    else:
        mark_gpu(False, ping.get("error") or "model not loaded")
        _status["reachable"] = bool(ping.get("reachable"))
        _status["cuda"] = bool(ping.get("cuda"))
    return gpu_status()


async def watch_gpu() -> None:
    """Background loop started with the server: keeps the saved GPU state fresh."""
    while True:
        try:
            await check_gpu()
        except Exception as error:
            mark_gpu(False, str(error)[:200])
        await asyncio.sleep(next_check_delay())
