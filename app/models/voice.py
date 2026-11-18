
from sqlalchemy import ForeignKey, Integer, String, Text, Boolean, DateTime, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.database.db import Base


class Voice(Base):
    __tablename__ = "voices"
    # One user can have many voices, but not two with the same name.
    __table_args__ = (UniqueConstraint("user_id", "name", name="voices_user_id_name_key"),)

    # "v_<10 hex>": the same id the recordings, stories.narrator_id and the TTS caches use.
    id: Mapped[str] = mapped_column(
        String(20), primary_key=True
    )

    # The owner (the logged-in user who recorded it): only this user lists, changes or narrates with it.
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    name: Mapped[str] = mapped_column(
        String(100), nullable=False
    )

    languages: Mapped[list[str]] = mapped_column(
        ARRAY(String(50)), nullable=False, default=list
    )

    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="Preparing"
    )

    is_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    avatar: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )

    # The original recording, relative to data/ (e.g. "recordings/v_x.m4a"). Copied to the bucket
    # with the prompt (app/services/voice_files.py).
    file_path: Mapped[str] = mapped_column(
        Text, nullable=False
    )

    prompt_path: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    voice_key: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )

    gpu_prepared: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
