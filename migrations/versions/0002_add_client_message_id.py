"""Add client_message_id column to message table.

Revision ID: 0002_add_client_message_id
Revises: 0001_initial
"""

from alembic import op

revision = "0002_add_client_message_id"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add the column if it doesn't already exist
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_schema = 'ai' AND table_name = 'message' AND column_name = 'client_message_id'
            ) THEN
                ALTER TABLE ai.message ADD COLUMN client_message_id UUID;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE ai.message DROP COLUMN IF EXISTS client_message_id")
