from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from pgvector.sqlalchemy import Vector
from sqlalchemy import event, text

from app.core.config import get_settings
from app.models.base import Base

settings = get_settings()


def get_database_url() -> str:
    if settings.database_url:
        return settings.database_url
    if settings.use_cloud_sql and settings.cloud_sql_instance:
        # Cloud SQL with asyncpg
        return f'postgresql+asyncpg://{settings.db_user}:{settings.db_pass}@/{settings.db_name}?host=/cloudsql/{settings.cloud_sql_instance}'
    # Local PostgreSQL
    return f'postgresql+asyncpg://{settings.db_user}:{settings.db_pass}@postgres:5432/{settings.db_name}'


engine = create_async_engine(
    get_database_url(),
    echo=settings.debug,
    poolclass=NullPool if settings.debug else None,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)


@event.listens_for(engine.sync_engine, 'connect')
def set_vector_extension(dbapi_connection, connection_record):
    '''Enable pgvector extension on connection'''
    try:
        with dbapi_connection.cursor() as cursor:
            cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
            dbapi_connection.commit()
    except Exception:
        pass


async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    '''Initialize database tables and indexes'''
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Create pgvector HNSW indexes after tables exist
        await create_vector_indexes(conn)


async def create_vector_indexes(conn) -> None:
    '''Create HNSW indexes for vector similarity search'''
    try:
        # Figures table - clip_embedding index
        await conn.execute(text('''
            CREATE INDEX IF NOT EXISTS idx_figures_clip_embedding_hnsw
            ON figures USING hnsw (clip_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        '''))
        # Shodhganga theses table
        await conn.execute(text('''
            CREATE INDEX IF NOT EXISTS idx_shodhganga_embeddings_hnsw
            ON shodhganga_theses USING hnsw (figure_embeddings vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        '''))
        await conn.commit()
    except Exception as e:
        # Index creation might fail if pgvector not installed or table doesn't exist yet
        pass


async def close_db() -> None:
    await engine.dispose()


@asynccontextmanager
async def lifespan_db():
    await init_db()
    try:
        yield
    finally:
        await close_db()
