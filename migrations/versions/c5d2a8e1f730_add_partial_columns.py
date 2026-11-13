"""add partial_key and partial_samples to story_audios

Revision ID: c5d2a8e1f730
Revises: 4a842bfb4d64
Create Date: 2026-11-12 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c5d2a8e1f730'
down_revision: Union[str, Sequence[str], None] = '4a842bfb4d64'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('story_audios', sa.Column('partial_key', sa.Text(), nullable=True))
    op.add_column('story_audios', sa.Column('partial_samples', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('story_audios', 'partial_samples')
    op.drop_column('story_audios', 'partial_key')
