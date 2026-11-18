from pydantic import BaseModel, EmailStr, Field, ConfigDict


class SignupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8, max_length=50)



class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email:EmailStr
    # No length rule here (that is for signup): a short password is just a wrong password.
    password:str = Field(min_length=1 ,max_length=128)



class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:int
    name:str
    email:EmailStr


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    # The signed-in user, so the app needs no extra /auth/me call.
    user: UserResponse


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    refresh_token: str



class TokenResponse(BaseModel):
    access_token:str
    token_type:str = "bearer"
