"""name users username unique constraint

Revision ID: c362125c26e7
Revises: 3099a3c3a476
Create Date: 2026-11-28 02:56:33.679099

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c362125c26e7"
down_revision: str | Sequence[str] | None = "3099a3c3a476"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Rename the username unique constraint."""
    op.execute(
        "ALTER TABLE users RENAME CONSTRAINT users_username_key TO uq_users_username"
    )


def downgrade() -> None:
    """Restore the original username unique constraint name."""
    op.execute(
        "ALTER TABLE users RENAME CONSTRAINT uq_users_username TO users_username_key"
    )
