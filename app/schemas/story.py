
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class StoryCreate(BaseModel):
    age: int | None = Field(default=None, ge=1, le=18)
    language: str = Field(default="Hindi", max_length=50)
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)
    story_type: str | None = Field(default=None, max_length=100)
    narrator_id: str | None = Field(default=None, max_length=100)


class StoryResponse(BaseModel):
    id: int
    age: int | None
    language: str
    title: str
    content: str
    story_type: str | None
    narrator_id: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
