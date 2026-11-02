"""Map document-wide chunks to page-local source ranges, preserving existing chunks."""

import sqlalchemy as sa
from alembic import op

revision = "0003_cross_page_chunks"
down_revision = "0002_document_chunks"
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    page_fk = sa.inspect(connection).get_foreign_keys("document_chunks")[0]["name"]
    op.drop_constraint(page_fk, "document_chunks", type_="foreignkey")
    op.drop_constraint("unique_page_chunk", "document_chunks", type_="unique")
    op.create_foreign_key(
        "document_chunks_document_id_fkey",
        "document_chunks",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint("unique_chunk_document", "document_chunks", ["id", "document_id"])
    op.create_table(
        "chunk_sources",
        sa.Column("chunk_id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), primary_key=True),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("chunk_start_char", sa.Integer(), nullable=False),
        sa.Column("chunk_end_char", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_id", "document_id"],
            ["document_chunks.id", "document_chunks.document_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id", "page_number"],
            ["document_pages.document_id", "document_pages.page_number"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("start_char >= 0 AND end_char > start_char", name="valid_source_range"),
        sa.CheckConstraint(
            "chunk_start_char >= 0 AND chunk_end_char > chunk_start_char",
            name="valid_source_chunk_range",
        ),
        sa.CheckConstraint(
            "end_char - start_char = chunk_end_char - chunk_start_char", name="source_range_length"
        ),
    )
    # Keep old page-local chunks usable until the document-v2 rebuild is run.
    op.execute("""
        INSERT INTO chunk_sources
            (chunk_id, document_id, page_number, start_char, end_char,
             chunk_start_char, chunk_end_char)
        SELECT id, document_id, page_number, start_char, end_char, 0, char_length(text)
        FROM document_chunks
    """)
    op.execute("""
        WITH page_offsets AS (
            SELECT document_id, page_number,
                COALESCE(SUM(char_length(text) + 1) OVER (
                    PARTITION BY document_id ORDER BY page_number
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS offset_chars
            FROM document_pages
        ), chunk_order AS (
            SELECT id, ROW_NUMBER() OVER (
                PARTITION BY document_id ORDER BY page_number, chunk_index
            ) - 1 AS new_index
            FROM document_chunks
        )
        UPDATE document_chunks AS c
        SET start_char = c.start_char + p.offset_chars,
            end_char = c.end_char + p.offset_chars,
            chunk_index = o.new_index
        FROM page_offsets AS p, chunk_order AS o
        WHERE c.document_id = p.document_id AND c.page_number = p.page_number AND c.id = o.id
    """)
    op.drop_column("document_chunks", "page_number")
    op.create_unique_constraint(
        "unique_document_chunk", "document_chunks", ["document_id", "chunk_index"]
    )
    # Configuration no longer describes the new input scope; the next run regenerates it.
    op.execute("UPDATE documents SET chunking_signature = NULL, chunking_config = NULL")


def downgrade():
    # A cross-page chunk cannot fit the old schema. Discard only this derived index.
    op.execute("DELETE FROM document_chunks")
    op.execute("UPDATE documents SET chunking_signature = NULL, chunking_config = NULL")
    op.drop_table("chunk_sources")
    op.drop_constraint("unique_document_chunk", "document_chunks", type_="unique")
    op.drop_constraint("unique_chunk_document", "document_chunks", type_="unique")
    op.drop_constraint("document_chunks_document_id_fkey", "document_chunks", type_="foreignkey")
    op.add_column("document_chunks", sa.Column("page_number", sa.Integer(), nullable=False))
    op.create_foreign_key(
        "document_chunks_document_id_page_number_fkey",
        "document_chunks",
        "document_pages",
        ["document_id", "page_number"],
        ["document_id", "page_number"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint(
        "unique_page_chunk", "document_chunks", ["document_id", "page_number", "chunk_index"]
    )
