from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db import get_db
from app.models.user import User

from app.core.security import ACCESS_TOKEN_TYPE, token_subject

bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized(detail: str = "Invalid authentication credentials") -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


async def user_from_subject(db: AsyncSession, sub: str) -> User | None:
    """The user a token's `sub` names: the user id, or the email in tokens made before that."""
    if sub.isdigit():
        return await db.get(User, int(sub))
    return await db.scalar(select(User).where(User.email == sub))


async def get_current_user(
        credentials:HTTPAuthorizationCredentials = Depends(bearer_scheme),
        db:AsyncSession =Depends(get_db)
):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()

    # Only an access token: a refresh token opens /auth/refresh and nothing else.
    sub = token_subject(credentials.credentials, ACCESS_TOKEN_TYPE)
    if sub is None:
        raise _unauthorized()

    user = await user_from_subject(db, sub)
    if user is None:
        raise _unauthorized("User not found")

    return user
