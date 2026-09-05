from logging.config import fileConfig
from sqlalchemy import pool, create_engine
from alembic import context
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.config import get_settings

settings = get_settings()

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None

def get_url():
    if settings.use_cloud_sql and settings.cloud_sql_instance:
        return f"postgresql+asyncpg://{settings.db_user}:{settings.db_pass}@/{settings.db_name}?host=/cloudsql/{settings.cloud_sql_instance}"
    return f"sqlite:///{settings.sqlite_path}"

def run_migrations_offline():
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = create_engine(get_url(), poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()