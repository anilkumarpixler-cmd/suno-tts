import jwt

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db import get_db
from app.models.user import User

from app.core.security import JWT_SECRET_KEY, JWT_ALGORITHM

bearer_scheme = HTTPBearer(auto_error=False)

async def get_current_user(
        credentials:HTTPAuthorizationCredentials = Depends(bearer_scheme),
        db:AsyncSession =Depends(get_db)
):
    if credentials is None  or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401,detail="Invalid authentication credentials",headers={"WWW-Authenticate":"Bearer"})


    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        user_email =payload.get("sub")

        if user_email is None:
            raise HTTPException(status_code=401,detail="Invalid authentication credentials",headers={"WWW-Authenticate":"Bearer"})


    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401,detail="Invalid authentication credentials",headers={"WWW-Authenticate":"Bearer"})


    user = await db.scalar(select(User).where(User.email == user_email))

    if user is None:
        raise HTTPException(status_code=401,detail="User not found",headers={"WWW-Authenticate":"Bearer"})


    return user;
    