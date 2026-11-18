import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_dependencies import get_current_user, user_from_subject
from app.core.rate_limit import limiter, login_failed, login_locked, login_ok

from app.database.db import get_db
from app.models.refresh_token import RefreshToken
from app.models.user import User

from app.schemas.auth import (
    SignupRequest,
    UserResponse,
    LoginRequest,
    LoginResponse,
    RefreshRequest,
)

from app.core.security import (
    REFRESH_TOKEN_EXPIRE_DAYS,
    REFRESH_TOKEN_TYPE,
    decode_token,
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
)


router =APIRouter(prefix="/auth", tags =["authentication"])

# Checked when the email is unknown, so the reply takes as long as a wrong password.
_DUMMY_HASH = hash_password("not-a-real-password")

# A revoked token sent again within this time is the app's own parallel refresh, not theft.
REUSE_GRACE_S = 30


def _login_again() -> HTTPException:
    return HTTPException(
        status_code=401, detail="Please log in again", headers={"WWW-Authenticate": "Bearer"}
    )


async def _tokens(user: User, db: AsyncSession) -> dict:
    """New access + refresh tokens. The refresh token's jti is saved, so it can be cancelled."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    jti = secrets.token_hex(16)
    # The user's old, expired entries are no use any more.
    await db.execute(
        delete(RefreshToken).where(RefreshToken.user_id == user.id, RefreshToken.expires_at < now)
    )
    db.add(RefreshToken(jti=jti, user_id=user.id, expires_at=expires_at))
    await db.flush()
    sub = str(user.id)
    return {
        "access_token": create_access_token(data={"sub": sub}),
        "refresh_token": create_refresh_token(data={"sub": sub, "jti": jti}, expires_at=expires_at),
        "token_type": "bearer",
        "user": user,
    }


@router.post("/signup", response_model=LoginResponse, status_code=201)
@limiter.limit("3/minute")
async def signup(request: Request, user: SignupRequest, db: AsyncSession = Depends(get_db)):
    # Saved as login compares it: lowercased, no spaces.
    email=str(user.email).strip().lower()
    name = user.name.strip()
    if len(name) < 3:
        raise HTTPException(status_code=400, detail="Name must be at least 3 characters")

    existing_user =await db.scalar(select(User).where(User.email == email))

    if existing_user:
        raise HTTPException(status_code=400, detail="Email Already Exists")

    new_user = User(
        name=name,
        email=email,
        password_hash=hash_password(user.password)
    )

    db.add(new_user)

    try:
        await db.commit()
        await db.refresh(new_user)
    except IntegrityError:
        # Two signups with the same email at once: the second one lands here.
        await db.rollback()
        raise HTTPException(status_code=400, detail="Email Already Exists")

    # Signed in straight away: tokens and the user in one reply.
    return await _tokens(new_user, db)


@router.post("/login", response_model=LoginResponse,status_code=200)
@limiter.limit("5/minute")
async def login(request: Request, user: LoginRequest, db: AsyncSession = Depends(get_db)):
    email=str(user.email).strip().lower()
    # Too many wrong passwords for this email, from any IP: wait before trying again.
    if login_locked(email):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in 15 minutes.")

    existing_user = await db.scalar(select(User).where(User.email == email))

    # One message for both cases: the reply never tells whether the email has an account.
    password_ok = verify_password(
        user.password, existing_user.password_hash if existing_user else _DUMMY_HASH
    )
    if existing_user is None or not password_ok:
        login_failed(email)
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    login_ok(email)
    return await _tokens(existing_user, db)


@router.post("/refresh", response_model=LoginResponse, status_code=200)
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    # The app calls this when its access token expired: new tokens, no password asked again.
    payload = decode_token(body.refresh_token, REFRESH_TOKEN_TYPE)
    if payload is None:
        raise _login_again()

    jti = payload.get("jti")
    if jti:
        # Each refresh token works once: it is cancelled here and a new one is returned.
        # Locked, so two refreshes with the same token can't both pass.
        saved = await db.scalar(
            select(RefreshToken).where(RefreshToken.jti == jti).with_for_update()
        )
        if saved is None:
            raise _login_again()
        now = datetime.now(timezone.utc)
        if saved.revoked:
            just_rotated = saved.revoked_at and saved.revoked_at > now - timedelta(seconds=REUSE_GRACE_S)
            if not just_rotated:
                # An old token came back: someone has a copy. Log this user out everywhere.
                await db.execute(
                    update(RefreshToken)
                    .where(RefreshToken.user_id == saved.user_id, RefreshToken.revoked.is_(False))
                    .values(revoked=True, revoked_at=now)
                )
                # Saved now: get_db rolls back when the 401 below is raised.
                await db.commit()
                print(f"[auth] refresh token reused for user {saved.user_id}: all sessions revoked", flush=True)
            raise _login_again()
        saved.revoked = True
        saved.revoked_at = now
    # No jti: made before refresh tokens were saved. Accepted once and swapped for a saved one,
    # so a login from before this change keeps working.

    user = await user_from_subject(db, payload["sub"])
    if user is None:
        raise _login_again()
    return await _tokens(user, db)


@router.post("/logout", status_code=204)
async def logout(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    # Cancels the refresh token; the short-lived access token simply runs out.
    # Always 204: logging out twice, or with an old token, is not an error.
    payload = decode_token(body.refresh_token, REFRESH_TOKEN_TYPE)
    jti = payload.get("jti") if payload else None
    if jti:
        saved = await db.get(RefreshToken, jti)
        if saved is not None and not saved.revoked:
            saved.revoked = True
            saved.revoked_at = datetime.now(timezone.utc)
    return Response(status_code=204)


@router.get("/me",response_model=UserResponse)
async def get_current_user_info(current_user:User =Depends(get_current_user)):
    return current_user
