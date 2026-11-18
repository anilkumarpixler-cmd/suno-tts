"""users.username -> name; voices belong to a user

Revision ID: d81f4c2b9a60
Revises: c5d2a8e1f730
Create Date: 2026-11-13 19:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd81f4c2b9a60'
down_revision: Union[str, Sequence[str], None] = 'c5d2a8e1f730'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # The person's name: names can repeat (the email is the login).
    op.drop_constraint('users_username_key', 'users', type_='unique')
    op.alter_column('users', 'username', new_column_name='name')
    # Login compares lowercased emails.
    op.execute("UPDATE users SET email = lower(trim(email))")

    # NULL: a voice saved on a phone before login; the first user who logs in there gets it.
    op.add_column('voices', sa.Column('user_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'voices_user_id_fkey', 'voices', 'users', ['user_id'], ['id'], ondelete='CASCADE'
    )
    op.create_index(op.f('ix_voices_user_id'), 'voices', ['user_id'], unique=False)
    # Two users on one phone may both have a "Mom": names are unique per user, not per phone.
    op.drop_constraint('voices_device_id_name_key', 'voices', type_='unique')
    op.create_unique_constraint('voices_user_id_name_key', 'voices', ['user_id', 'name'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('voices_user_id_name_key', 'voices', type_='unique')
    op.create_unique_constraint('voices_device_id_name_key', 'voices', ['device_id', 'name'])
    op.drop_index(op.f('ix_voices_user_id'), table_name='voices')
    op.drop_constraint('voices_user_id_fkey', 'voices', type_='foreignkey')
    op.drop_column('voices', 'user_id')
    op.alter_column('users', 'name', new_column_name='username')
    op.create_unique_constraint('users_username_key', 'users', ['username'])
