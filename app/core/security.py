import os
from datetime import datetime ,timedelta ,timezone

import jwt
from pwdlib import PasswordHash

password_hash = PasswordHash.recommended()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing in .env")

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM","HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES",10))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS",7))

# A refresh token only gets new tokens (/auth/refresh); it never opens other routes.
ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"


def decode_token(token: str, token_type: str) -> dict | None:
    """The claims of a valid token of `token_type` that has a subject, else None. Tokens made
    before the type claim count as access tokens, so a saved login keeps working."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        return None
    if payload.get("type", ACCESS_TOKEN_TYPE) != token_type:
        return None
    sub = payload.get("sub")
    return payload if isinstance(sub, str) and sub else None


def token_subject(token: str, token_type: str) -> str | None:
    """The `sub` of a valid token: the user id (older tokens: the email)."""
    payload = decode_token(token, token_type)
    return payload["sub"] if payload else None


def hash_password(password:str) -> str:
    return password_hash.hash(password)


def verify_password(password:str ,hashed_password:str) ->bool:
    return password_hash.verify(password ,hashed_password)


def create_access_token(data:dict ,expires_delta:timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp":expire, "type":ACCESS_TOKEN_TYPE})
    encoded_jwt = jwt.encode(to_encode ,JWT_SECRET_KEY ,algorithm=JWT_ALGORITHM)
    return encoded_jwt

def create_refresh_token(data:dict ,expires_at:datetime | None = None) -> str:
    """`data` carries `sub` and the `jti` saved in refresh_tokens (so it can be cancelled)."""
    to_encode = data.copy()
    expire = expires_at or datetime.now(timezone.utc) + timedelta(minutes=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp":expire, "type":REFRESH_TOKEN_TYPE})
    encoded_jwt = jwt.encode(to_encode ,JWT_SECRET_KEY ,algorithm=JWT_ALGORITHM)
    return encoded_jwt
