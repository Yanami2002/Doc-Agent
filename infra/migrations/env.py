"""Run migrations with the same configuration as the application."""

from alembic import context

from jp_doc_agent.config import Settings
from jp_doc_agent.database import create_database_engine
from jp_doc_agent.models import Base


def migrate(connection):
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        version_table_schema=context.config.attributes.get("version_table_schema"),
    )
    with context.begin_transaction():
        context.run_migrations()


connection = context.config.attributes.get("connection")
if connection is not None:
    migrate(connection)
else:
    engine = create_database_engine(Settings())
    try:
        with engine.connect() as connection:
            migrate(connection)
    finally:
        engine.dispose()
