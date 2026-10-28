
from sqlalchemy import String, Text, Integer, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.db import Base


class Story(Base):
    __tablename__ = "stories"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    child_name: Mapped[str] = mapped_column(
        String(100), nullable=True
    )

    age: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    language: Mapped[str] = mapped_column(
        String(50), nullable=False, default="Hindi"
    )

    title: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )

    content: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    story_type: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )

    narrator_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )

    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="pending"
    )

    audio_url: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    duration_seconds: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
