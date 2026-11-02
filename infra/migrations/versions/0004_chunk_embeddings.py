"""チャンクに対応する Embedding 設定と 1536 次元のベクトルを保存する。"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR

revision = "0004_chunk_embeddings"
down_revision = "0003_cross_page_chunks"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "embedding_profiles",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("dimensions = 1536", name="embedding_profile_dimensions"),
    )
    op.create_table(
        "chunk_embeddings",
        sa.Column(
            "chunk_id",
            sa.Integer(),
            sa.ForeignKey("document_chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "profile_id", sa.String(64), sa.ForeignKey("embedding_profiles.id"), primary_key=True
        ),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", VECTOR(1536), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("token_count > 0 AND token_count <= 300", name="embedding_token_count"),
        sa.CheckConstraint("vector_norm(embedding) > 0", name="nonzero_embedding"),
    )


def downgrade():
    op.drop_table("chunk_embeddings")
    op.drop_table("embedding_profiles")
