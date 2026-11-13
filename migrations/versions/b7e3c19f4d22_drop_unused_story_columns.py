"""drop unused story columns

Revision ID: b7e3c19f4d22
Revises: a4d91b2a0413
Create Date: 2026-10-29 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7e3c19f4d22'
down_revision: Union[str, Sequence[str], None] = 'a4d91b2a0413'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column('stories', 'child_name')
    op.drop_column('stories', 'status')
    op.drop_column('stories', 'audio_url')
    op.drop_column('stories', 'duration_seconds')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column('stories', sa.Column('duration_seconds', sa.Integer(), nullable=True))
    op.add_column('stories', sa.Column('audio_url', sa.Text(), nullable=True))
    op.add_column('stories', sa.Column('status', sa.String(length=30), server_default='pending', nullable=False))
    op.add_column('stories', sa.Column('child_name', sa.String(length=100), nullable=True))
