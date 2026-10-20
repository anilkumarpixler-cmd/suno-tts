from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import httpx

from app.config import COLAB_TTS_SECRET, COLAB_TTS_TIMEOUT, COLAB_TTS_URL
from app.services.gpu_status import mark_gpu


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


# One client for all Kaggle calls, so chunks reuse the open connection through the tunnel
# instead of a new TCP/TLS handshake each time.
_client: httpx.AsyncClient | None = None


def _shared_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=COLAB_TTS_TIMEOUT,
            follow_redirects=False,
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=60.0),
        )
    return _client


async def _close_later(client: httpx.AsyncClient) -> None:
    # Other chunks may still be using the old client: close it once they are done.
    await asyncio.sleep(COLAB_TTS_TIMEOUT)
    await client.aclose()


def _reset_client(failed: httpx.AsyncClient) -> None:
    """Kaggle restarted or the tunnel changed: the next call opens a new connection."""
    global _client
    if _client is failed:
        _client = None
        asyncio.get_running_loop().create_task(_close_later(failed))
        print("[colab] connection reset, next call opens a new one", flush=True)


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


# sha256 of each prompt file, keyed by (path, mtime, size), so a file is hashed once, not per chunk.
_voice_keys: dict[tuple[str, int, int], str] = {}


def _voice_key(prompt_path: str) -> str:
    # Same key the Kaggle server gives a stored voice: sha256 of the prompt file bytes.
    stat = Path(prompt_path).stat()
    cache_key = (prompt_path, stat.st_mtime_ns, stat.st_size)
    key = _voice_keys.get(cache_key)
    if key is None:
        key = hashlib.sha256(Path(prompt_path).read_bytes()).hexdigest()
        _voice_keys[cache_key] = key
    return key


def _raise_for_colab(response: httpx.Response, action: str) -> None:
    if response.status_code in {301, 302, 303, 307, 308}:
        raise RuntimeError(f"Colab returned a redirect interstitial on {action}")
    if response.status_code in {502, 503}:
        detail = (response.text or response.reason_phrase or "")[:300]
        raise RuntimeError(_tunnel_down_message(response.status_code, detail))
    if response.status_code >= 400:
        detail = (response.text or response.reason_phrase or "")[:300]
        raise RuntimeError(f"Colab {action} failed ({response.status_code}): {detail}")


def _is_unknown_voice(response: httpx.Response) -> bool:
    return response.status_code == 404 and "voice not found" in (response.text or "").lower()


async def _post_generate(
    client: httpx.AsyncClient,
    text: str,
    language_id: str,
    voice_key: str | None = None,
    prompt_path: str | None = None,
) -> httpx.Response:
    data = {"text": text, "language_id": language_id}
    if voice_key:
        data["voice_key"] = voice_key
    if prompt_path is None:
        return await client.post(f"{COLAB_TTS_URL}/generate", data=data, headers=_colab_headers())
    with open(prompt_path, "rb") as handle:
        files = {"audio": (Path(prompt_path).name, handle, "application/octet-stream")}
        return await client.post(
            f"{COLAB_TTS_URL}/generate", data=data, files=files, headers=_colab_headers()
        )


def _conds_path(prompt_path: str, voice_key: str) -> Path:
    # Prepared voice (Chatterbox Conditionals) saved next to the prompt, so a Kaggle restart
    # can restore it without preparing the voice again.
    prompt = Path(prompt_path)
    return prompt.with_name(f"{prompt.name}.{voice_key[:16]}.conds.pt")


async def _download_conds(client: httpx.AsyncClient, prompt_path: str, voice_key: str) -> None:
    # Best effort: older servers have no conds endpoint, playback still works without it.
    try:
        response = await client.get(f"{COLAB_TTS_URL}/voice/{voice_key}/conds", headers=_colab_headers())
        if response.status_code != 200 or not response.content or _is_html(
            response.headers.get("content-type") or "", response.content
        ):
            return
        _conds_path(prompt_path, voice_key).write_bytes(response.content)
        print(f"[colab] saved conds {voice_key[:8]}", flush=True)
    except Exception as error:
        print(f"[colab] conds download failed {voice_key[:8]}: {error}", flush=True)


async def _restore_conds(client: httpx.AsyncClient, prompt_path: str, voice_key: str) -> bool:
    conds = _conds_path(prompt_path, voice_key)
    if not conds.exists():
        return False
    try:
        with open(conds, "rb") as handle:
            files = {"conds": (conds.name, handle, "application/octet-stream")}
            response = await client.post(
                f"{COLAB_TTS_URL}/voice_conds",
                data={"voice_key": voice_key},
                files=files,
                headers=_colab_headers(),
            )
    except Exception as error:
        print(f"[colab] conds restore failed {voice_key[:8]}: {error}", flush=True)
        return False
    if response.status_code != 200:
        print(f"[colab] conds restore failed {voice_key[:8]} ({response.status_code})", flush=True)
        return False
    print(f"[colab] restored voice {voice_key[:8]} from conds", flush=True)
    return True


async def _upload_voice(client: httpx.AsyncClient, prompt_path: str) -> str:
    with open(prompt_path, "rb") as handle:
        files = {"audio": (Path(prompt_path).name, handle, "application/octet-stream")}
        response = await client.post(f"{COLAB_TTS_URL}/voice", files=files, headers=_colab_headers())
    _raise_for_colab(response, "voice upload")
    try:
        voice_key = str(response.json()["voice_key"])
    except Exception as error:
        raise RuntimeError("Colab voice upload did not return a voice_key") from error
    if not _conds_path(prompt_path, voice_key).exists():
        await _download_conds(client, prompt_path, voice_key)
    return voice_key


async def upload_voice_to_gpu(prompt_path: str) -> str:
    """Upload a prompt once at save time; the GPU server prepares and keeps its conds."""
    if not COLAB_TTS_URL:
        raise RuntimeError("COLAB_TTS_URL is not set")
    client = _shared_client()
    voice_key = _voice_key(prompt_path)
    if await _restore_conds(client, prompt_path, voice_key):
        return voice_key
    return await _upload_voice(client, prompt_path)


async def _generate(client: httpx.AsyncClient, text: str, language_id: str, prompt_path: str) -> httpx.Response:
    # Send only the voice key; the server keeps the prompt from the first upload.
    voice_key = _voice_key(prompt_path)
    response = await _post_generate(client, text, language_id, voice_key=voice_key)
    if _is_unknown_voice(response):
        # Kaggle session restarted: send the saved prepared voice so it is not prepared again.
        # First use of this voice (no saved conds): upload the prompt once.
        if not await _restore_conds(client, prompt_path, voice_key):
            print(f"[colab] uploading voice {voice_key[:8]}", flush=True)
            voice_key = await _upload_voice(client, prompt_path)
        response = await _post_generate(client, text, language_id, voice_key=voice_key)
    elif response.status_code == 422:
        # Older server (Colab) without voice_key support: send the prompt file every chunk.
        response = await _post_generate(client, text, language_id, prompt_path=prompt_path)
    return response


async def generate_on_colab(text: str, language_id: str, prompt_path: str) -> bytes:
    if not COLAB_TTS_URL:
        raise RuntimeError("COLAB_TTS_URL is not set")

    client = _shared_client()
    try:
        response = await _generate(client, text, language_id, prompt_path)
    except httpx.TransportError as error:
        # The kept-open connection died (Kaggle restarted / tunnel dropped): retry once on a new one.
        print(f"[colab] connection failed ({type(error).__name__}), retrying on a new connection", flush=True)
        _reset_client(client)
        client = _shared_client()
        try:
            response = await _generate(client, text, language_id, prompt_path)
        except httpx.TransportError as retry_error:
            _reset_client(client)
            # Real chunks count as health checks: the app hears "GPU down" without waiting.
            mark_gpu(False, f"connection failed ({type(retry_error).__name__})")
            raise
    if response.status_code in {502, 503}:
        # Tunnel down: don't keep this connection for the next chunk.
        _reset_client(client)
        mark_gpu(False, _tunnel_down_message(response.status_code, ""))

    content_type = (response.headers.get("content-type") or "")[:80]
    body = response.content or b""

    _raise_for_colab(response, "generate")
    if _is_html(content_type, body):
        mark_gpu(False, "HTML interstitial instead of audio")
        raise RuntimeError("Colab returned HTML interstitial, not audio")
    if not body:
        raise RuntimeError("Colab returned empty audio")
    if "audio/" not in content_type.lower() and not _looks_like_wav(body):
        raise RuntimeError(f"Colab returned non-audio content ({content_type or 'unknown'})")
    mark_gpu(True)
    return body
