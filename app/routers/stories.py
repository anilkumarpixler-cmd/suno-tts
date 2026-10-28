
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db import get_db
from app.models.story import Story
from app.schemas.story import StoryCreate, StoryResponse

router = APIRouter(prefix="/stories", tags=["Stories"])


# Create a story
@router.post("/", response_model=StoryResponse, status_code=201)
async def create_story(
    story_data: StoryCreate,
    db: AsyncSession = Depends(get_db),
):
    new_story = Story(
        # id=story_data.id,
        child_name=story_data.child_name,
        age=story_data.age,
        language=story_data.language,
        title=story_data.title,
        content=story_data.content,
        story_type=story_data.story_type,
        narrator_id=story_data.narrator_id,
        status="pending",
    )

    try:
        db.add(new_story)
        await db.commit()
        await db.refresh(new_story)
        return new_story
    except Exception:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to save the story.",
        )


# Get all stories
@router.get("/", response_model=list[StoryResponse])
async def list_stories(
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Story).order_by(Story.created_at.desc())
    )
    return result.scalars().all()


# Get one story by ID
@router.get("/{story_id}", response_model=StoryResponse)
async def get_story(
    story_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Story).where(Story.id == story_id)
    )
    story = result.scalar_one_or_none()

    if story is None:   
        raise HTTPException(
            status_code=404,
            detail="Story not found.",
        )

    return story
