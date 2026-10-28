from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_dependencies import get_current_user

from app.database.db import get_db
from app.models.user import User

from app.schemas.auth import (
    SignupRequest,
    SignupResponse,
    LoginRequest,
    LoginResponse
)

from app.core.security import (
    hash_password,      
    verify_password,
    create_access_token,
    create_refresh_token,
)


router =APIRouter(prefix="/auth", tags =["authentication"])

@router.post("/signup", response_model=SignupResponse, status_code=201)
async def signup(user:SignupRequest,db:AsyncSession = Depends(get_db)):
    email=str(user.email).strip().lower()

    existing_user =await db.scalar(select(User).where(User.email == email))

    if existing_user:
        raise HTTPException(status_code=400, detail="Email Already Exists")

    new_user = User(
    username=user.username,
    email=user.email,
    password_hash=hash_password(user.password)
    )


    db.add(new_user)

    try:
        await db.commit()
        await db.refresh(new_user)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=400, detail="Email Already Exists")


    return new_user


@router.post("/login", response_model=LoginResponse,status_code=200)
async def login(user:LoginRequest,db:AsyncSession = Depends(get_db)):
    email=str(user.email).strip().lower()

    existing_user = await db.scalar(select(User).where(User.email == email))

    if not existing_user:
        raise HTTPException(status_code=401, detail="Invalid Email")

    if not verify_password(user.password,existing_user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect Password",headers={"WWW-Authenticate":"Bearer"})


    access_token = create_access_token(data={"sub":existing_user.email})
    refresh_token = create_refresh_token(data={"sub":existing_user.email})

    return {
        "access_token":access_token,
        "refresh_token":refresh_token,
        "token_type":"bearer" 
    }


@router.get("/me",response_model=SignupResponse)
async def get_current_user_info(current_user:User =Depends(get_current_user)):
    return current_user