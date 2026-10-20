"""Initial AI service schema.

Revision ID: 0001_initial
Revises:
"""

from alembic import op

from ai.db.models import Base

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
    op.execute("CREATE SCHEMA IF NOT EXISTS ai")
    op.execute(
        "CREATE TEXT SEARCH CONFIGURATION ai.fts_unaccent (COPY = pg_catalog.simple)"
    )
    op.execute(
        "ALTER TEXT SEARCH CONFIGURATION ai.fts_unaccent "
        "ALTER MAPPING FOR hword, hword_part, word WITH unaccent, simple"
    )
    Base.metadata.create_all(bind=op.get_bind(), checkfirst=False)


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind(), checkfirst=False)
    op.execute("DROP TEXT SEARCH CONFIGURATION ai.fts_unaccent")
    op.execute("DROP SCHEMA ai")
