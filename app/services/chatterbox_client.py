from __future__ import annotations

import httpx

from app.config import COLAB_TTS_SECRET, COLAB_TTS_TIMEOUT, COLAB_TTS_URL


def colab_configured() -> bool:
    return bool(COLAB_TTS_URL)


def _uses_localtunnel() -> bool:
    host = COLAB_TTS_URL.lower()
    return "loca.lt" in host or "localtunnel" in host


def _colab_headers() -> dict[str, str]:
    headers = {"Accept": "application/json, audio/wav, audio/*, */*"}
    if COLAB_TTS_SECRET:
        headers["X-TTS-Secret"] = COLAB_TTS_SECRET
    if _uses_localtunnel():
        headers["bypass-tunnel-reminder"] = "1"
        headers["User-Agent"] = "suno-tts/1.0"
    return headers


def _tunnel_down_message(status: int, detail: str) -> str:
    return f"Colab tunnel down ({status}): {detail or 'Bad Gateway / Tunnel Unavailable'}"


def _is_html(content_type: str, body: bytes) -> bool:
    lowered = (content_type or "").lower()
    if "text/html" in lowered:
        return True
    head = body[:200].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def _looks_like_wav(body: bytes) -> bool:
    return len(body) >= 12 and body[:4] == b"RIFF" and body[8:12] == b"WAVE"


async def ping_colab() -> dict:
    if not COLAB_TTS_URL:
        return {
            "reachable": False,
            "cuda": False,
            "loaded": False,
            "error": "COLAB_TTS_URL is not set",
        }
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=False) as client:
            response = await client.get(f"{COLAB_TTS_URL}/health", headers=_colab_headers())
    except Exception as error:
        return {
            "reachable": False,
            "cuda": False,
            "loaded": False,
            "error": str(error)[:200],
        }

    if response.status_code in {301, 302, 303, 307, 308}:
        return {
            "reachable": False,
            "cuda": False,
            "loaded": False,
            "error": "redirect interstitial (tunnel reminder page)",
        }
    if response.status_code in {502, 503}:
        return {
            "reachable": False,
            "cuda": False,
            "loaded": False,
            "error": _tunnel_down_message(response.status_code, (response.text or "")[:120]),
        }
    if _is_html(response.headers.get("content-type") or "", response.content or b""):
        return {
            "reachable": False,
            "cuda": False,
            "loaded": False,
            "error": "HTML interstitial instead of /health JSON",
        }
    if response.status_code >= 400:
        return {
            "reachable": False,
            "cuda": False,
            "loaded": False,
            "error": f"health {response.status_code}: {(response.text or '')[:120]}",
        }
    try:
        data = response.json()
    except Exception:
        return {
            "reachable": False,
            "cuda": False,
            "loaded": False,
            "error": "health response was not JSON",
        }
    return {
        "reachable": True,
        "cuda": bool(data.get("cuda")),
        "loaded": bool(data.get("loaded")),
        "error": None,
    }


async def generate_on_colab(text: str, language_id: str, prompt_path: str) -> bytes:
    if not COLAB_TTS_URL:
        raise RuntimeError("COLAB_TTS_URL is not set")

    headers = _colab_headers()
    filename = prompt_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    with open(prompt_path, "rb") as handle:
        files = {"audio": (filename, handle, "application/octet-stream")}
        data = {"text": text, "language_id": language_id}
        async with httpx.AsyncClient(timeout=COLAB_TTS_TIMEOUT, follow_redirects=False) as client:
            response = await client.post(
                f"{COLAB_TTS_URL}/generate",
                data=data,
                files=files,
                headers=headers,
            )

    content_type = (response.headers.get("content-type") or "")[:80]
    body = response.content or b""

    if response.status_code in {301, 302, 303, 307, 308}:
        raise RuntimeError("Colab returned a redirect interstitial, not audio")
    if response.status_code in {502, 503}:
        detail = (response.text or response.reason_phrase or "")[:300]
        raise RuntimeError(_tunnel_down_message(response.status_code, detail))
    if response.status_code >= 400:
        detail = (response.text or response.reason_phrase or "")[:300]
        raise RuntimeError(f"Colab generate failed ({response.status_code}): {detail}")
    if _is_html(content_type, body):
        raise RuntimeError("Colab returned HTML interstitial, not audio")
    if not body:
        raise RuntimeError("Colab returned empty audio")
    if "audio/" not in content_type.lower() and not _looks_like_wav(body):
        raise RuntimeError(f"Colab returned non-audio content ({content_type or 'unknown'})")
    return body
