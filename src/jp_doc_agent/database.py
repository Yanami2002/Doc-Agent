"""Database connection and a read-only PostgreSQL/pgvector check."""

from sqlalchemy import Engine, create_engine, text

from jp_doc_agent.config import Settings


def create_database_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5, "options": "-c statement_timeout=5000"},
    )


def check_database(engine: Engine) -> dict[str, str | float]:
    """Verify connectivity, the extension, and an actual vector operation."""
    with engine.connect() as connection:
        postgres_version = connection.execute(text("SHOW server_version")).scalar_one()
        vector_version = connection.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()
        if vector_version is None:
            raise RuntimeError(
                "vector 拡張が有効ではありません。infra/docker/init.sql を実行してください。"
            )

        distance = connection.execute(
            text("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector")
        ).scalar_one()
        if distance != 1.0:
            raise RuntimeError("ベクトル距離の計算結果が想定と異なります。")

    return {
        "postgresql": postgres_version,
        "pgvector": vector_version,
        "vector_distance": float(distance),
    }
