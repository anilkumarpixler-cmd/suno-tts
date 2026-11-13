from sqlalchemy import (
    String, Text, Integer, SmallInteger, DateTime,
    ForeignKey, CheckConstraint, Index, func, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database.db import Base


class StoryAudio(Base):
    __tablename__ = "story_audios"
    __table_args__ = (
        CheckConstraint(
            "status IN ('generating','partial','ready','failed','cancelled')",
            name="ck_story_audios_status",
        ),
        Index("ix_story_audios_device", "device_id", text("created_at DESC")),
        # Small index for resume and cleanup jobs.
        Index(
            "ix_story_audios_pending", "updated_at",
            postgresql_where=text("status IN ('generating','partial')"),
        ),
        {"postgresql_with": {"fillfactor": 90}},
    )

    # sha256 prefix of story + voice + language, from story_stream.py.
    story_key: Mapped[str] = mapped_column(
        String(32), primary_key=True
    )

    story_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("stories.id", ondelete="CASCADE"), nullable=True, index=True
    )

    voice_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("voices.id", ondelete="CASCADE"), nullable=False, index=True
    )

    language: Mapped[str] = mapped_column(
        String(50), nullable=False
    )

    device_id: Mapped[str] = mapped_column(
        String(64), nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(12), nullable=False, server_default="generating"
    )

    chunk_count: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="0"
    )

    chunks_done: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="0"
    )

    # Saved only when partial; NULL once ready.
    chunks: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True
    )

    # Relative to data/tts/ (e.g. "story_streams/ab/<key>/final.wav"); the audio stays on disk.
    final_key: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    # A partial story left long enough: its chunks packed into tmp/<key>/partial.flac in the bucket
    # (local folder removed), and the frame count of each chunk to split it back. NULL otherwise.
    partial_key: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    partial_samples: Mapped[list[int] | None] = mapped_column(
        JSONB, nullable=True
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    position_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )

    last_chunk_index: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="0"
    )

    last_played_at: Mapped[DateTime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
