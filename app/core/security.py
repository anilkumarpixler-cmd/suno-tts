import os
from datetime import datetime ,timedelta ,timezone

import jwt
from pwdlib import PasswordHash

password_hash = PasswordHash.recommended()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM","HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES",10))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS",7))


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
    to_encode.update({"exp":expire})
    encoded_jwt = jwt.encode(to_encode ,JWT_SECRET_KEY ,algorithm=JWT_ALGORITHM)
    return encoded_jwt

def create_refresh_token(data:dict ,expires_delta:timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp":expire})
    encoded_jwt = jwt.encode(to_encode ,JWT_SECRET_KEY ,algorithm=JWT_ALGORITHM)
    return encoded_jwt