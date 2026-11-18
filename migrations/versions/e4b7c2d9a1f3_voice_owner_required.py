"""voice owner required, drop voices.device_id

Revision ID: e4b7c2d9a1f3
Revises: 78f78dbd3486
Create Date: 2026-11-16 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e4b7c2d9a1f3'
down_revision: Union[str, Sequence[str], None] = '78f78dbd3486'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Every voice belongs to a user; the phone id no longer decides the owner."""
    orphans = op.get_bind().execute(sa.text("SELECT id FROM voices WHERE user_id IS NULL")).scalars().all()
    if orphans:
        raise RuntimeError(
            f"Voices without an owner: {', '.join(orphans)}. Delete them or set user_id before upgrading."
        )
    op.alter_column('voices', 'user_id', existing_type=sa.Integer(), nullable=False)
    op.drop_index('ix_voices_device_id', table_name='voices')
    op.drop_column('voices', 'device_id')


def downgrade() -> None:
    """The phone ids are not restored (empty)."""
    op.add_column('voices', sa.Column('device_id', sa.String(length=64), nullable=False, server_default=''))
    op.create_index('ix_voices_device_id', 'voices', ['device_id'], unique=False)
    op.alter_column('voices', 'user_id', existing_type=sa.Integer(), nullable=True)
