"""Add page-local chunks and reproducible chunking configuration."""

import sqlalchemy as sa
from alembic import op

revision = "0002_document_chunks"
down_revision = "0001_documents"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("documents", sa.Column("chunking_signature", sa.String(64), nullable=True))
    op.add_column("documents", sa.Column("chunking_config", sa.JSON(), nullable=True))
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id", "page_number"],
            ["document_pages.document_id", "document_pages.page_number"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("document_id", "page_number", "chunk_index", name="unique_page_chunk"),
        sa.CheckConstraint("chunk_index >= 0", name="nonnegative_chunk_index"),
        sa.CheckConstraint("start_char >= 0 AND end_char > start_char", name="valid_chunk_range"),
        sa.CheckConstraint("char_length(text) = end_char - start_char", name="chunk_text_length"),
    )


def downgrade():
    op.drop_table("document_chunks")
    op.drop_column("documents", "chunking_config")
    op.drop_column("documents", "chunking_signature")
