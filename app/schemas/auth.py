from pydantic import BaseModel, EmailStr, Field, ConfigDict


class SignupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8, max_length=50)



class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email:EmailStr
    password:str = Field(min_length=8 ,max_length=50)



class SignupResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:int
    email:EmailStr


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"



class TokenResponse(BaseModel):
    access_token:str
    token_type:str = "bearer"
