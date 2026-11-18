import os
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

# Only these addresses (the tunnel on this machine) may say who the real caller is. A header from
# anyone else is ignored, so a caller can't send a fresh fake IP with every attempt.
TRUSTED_PROXIES = {
    ip.strip() for ip in os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1").split(",") if ip.strip()
}


def client_ip(request: Request) -> str:
    """The caller's IP. Behind the tunnel every request comes from 127.0.0.1, so there the address
    the tunnel forwards counts; a direct caller is counted by its real connection address."""
    peer = request.client.host if request.client else "unknown"
    if peer in TRUSTED_PROXIES:
        forwarded = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return peer


# Counts are in memory: reset on restart and kept per uvicorn process (run one worker).
limiter = Limiter(key_func=client_ip)


def rate_limit_exceeded(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    # `detail`, like every other error, so the app shows this line to the user.
    return JSONResponse(
        status_code=429, content={"detail": "Too many attempts. Please wait a minute and try again."}
    )


# Failed logins per email, whatever IP they come from: after 5 within 15 minutes the email is
# locked until the oldest of them is 15 minutes old.
LOGIN_MAX_FAILS = 5
LOGIN_LOCK_SECONDS = 15 * 60
_login_fails: dict[str, list[float]] = {}


def login_locked(email: str) -> bool:
    now = time.time()
    recent = [t for t in _login_fails.get(email, []) if now - t < LOGIN_LOCK_SECONDS]
    if recent:
        _login_fails[email] = recent
    else:
        # Nothing recent: drop the entry so the dict doesn't grow with every email ever tried.
        _login_fails.pop(email, None)
    return len(recent) >= LOGIN_MAX_FAILS


def login_failed(email: str) -> None:
    _login_fails.setdefault(email, []).append(time.time())


def login_ok(email: str) -> None:
    _login_fails.pop(email, None)
