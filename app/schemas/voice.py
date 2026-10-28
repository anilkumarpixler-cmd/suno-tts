
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class VoicePatch(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    is_default: bool | None = None
    avatar: str | None = Field(default=None, max_length=20)


class VoiceJson(BaseModel):
    name: str = Field(max_length=100)
    languages: str | list[str] = "English"
    audio_base64: str
    filename: str = "recording.m4a"


class VoiceResponse(BaseModel):
    id: str
    name: str
    languages: list[str]
    status: str
    is_default: bool
    avatar: str | None
    gpu_prepared: bool
    created_at: datetime
    # Same keys the app read from the voices.json version of this API.
    isDefault: bool
    audio_url: str
    audioUri: str

    model_config = ConfigDict(from_attributes=True)
