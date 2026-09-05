import sqlite3
from typing import Optional

try:
    import asyncpg
    from google.cloud.sql.connector import Connector, IPTypes
    CLOUD_SQL_AVAILABLE = True
except ImportError:
    CLOUD_SQL_AVAILABLE = False
    Connector = None
    IPTypes = None

from core.config import get_settings

settings = get_settings()

USE_CLOUD_SQL = settings.use_cloud_sql and CLOUD_SQL_AVAILABLE
INSTANCE_CONNECTION_NAME = settings.cloud_sql_instance
DB_USER = settings.db_user
DB_PASS = settings.db_pass
DB_NAME = settings.db_name
PRIVATE_IP = settings.private_ip
SQLITE_PATH = settings.sqlite_path

_connector: Optional[Connector] = None
_pool: Optional["asyncpg.Pool"] = None


async def get_connector() -> Optional[Connector]:
    global _connector
    if not USE_CLOUD_SQL or not INSTANCE_CONNECTION_NAME:
        return None
    if _connector is None:
        _connector = Connector()
    return _connector


async def get_pool() -> Optional[asyncpg.Pool]:
    global _pool
    if _pool is not None:
        return _pool

    connector = await get_connector()
    if connector is None:
        return None

    ip_type = IPTypes.PRIVATE if PRIVATE_IP else IPTypes.PUBLIC

    _pool = await connector.create_asyncpg_pool(
        instance_connection_string=INSTANCE_CONNECTION_NAME,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        min_size=1,
        max_size=10,
        ip_type=ip_type,
        command_timeout=60,
    )
    return _pool


async def get_cloud_sql_conn() -> Optional[asyncpg.Connection]:
    pool = await get_pool()
    if pool is None:
        return None
    return await pool.acquire()


async def release_cloud_sql_conn(conn: asyncpg.Connection):
    pool = await get_pool()
    if pool and conn:
        await pool.release(conn)


def get_sqlite_conn():
    return sqlite3.connect(SQLITE_PATH)


async def init_db():
    if USE_CLOUD_SQL and INSTANCE_CONNECTION_NAME:
        conn = await get_cloud_sql_conn()
        if conn:
            try:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS figure_hashes (
                        id SERIAL PRIMARY KEY,
                        paper_name TEXT NOT NULL,
                        figure_path TEXT NOT NULL,
                        phash TEXT NOT NULL,
                        dhash TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                ''')
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_figure_hashes_phash 
                    ON figure_hashes (phash)
                ''')
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_figure_hashes_paper 
                    ON figure_hashes (paper_name)
                ''')
            finally:
                await release_cloud_sql_conn(conn)
            return

    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS figure_hashes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_name TEXT NOT NULL,
            figure_path TEXT NOT NULL,
            phash TEXT NOT NULL,
            dhash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_figure_hashes_phash ON figure_hashes (phash)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_figure_hashes_paper ON figure_hashes (paper_name)')
    conn.commit()
    conn.close()


async def save_figure_hash(paper_name: str, figure_path: str, phash: str, dhash: str):
    if USE_CLOUD_SQL and INSTANCE_CONNECTION_NAME:
        conn = await get_cloud_sql_conn()
        if conn:
            try:
                await conn.execute(
                    'INSERT INTO figure_hashes (paper_name, figure_path, phash, dhash) VALUES ($1, $2, $3, $4)',
                    paper_name, figure_path, phash, dhash
                )
                return
            finally:
                await release_cloud_sql_conn(conn)

    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO figure_hashes (paper_name, figure_path, phash, dhash) VALUES (?, ?, ?, ?)',
        (paper_name, figure_path, phash, dhash)
    )
    conn.commit()
    conn.close()


def calculate_hamming_distance(hash1_hex: str, hash2_hex: str) -> int:
    val1 = int(hash1_hex, 16)
    val2 = int(hash2_hex, 16)
    return bin(val1 ^ val2).count('1')


async def find_matches(
    query_phash: str, 
    query_dhash: str, 
    threshold: Optional[int] = None,
    exclude_paper: Optional[str] = None
):
    threshold = threshold or settings.hybrid_threshold
    rows = []

    if USE_CLOUD_SQL and INSTANCE_CONNECTION_NAME:
        conn = await get_cloud_sql_conn()
        if conn:
            try:
                if exclude_paper:
                    rows = await conn.fetch(
                        'SELECT paper_name, figure_path, phash, dhash FROM figure_hashes WHERE paper_name != $1',
                        exclude_paper
                    )
                else:
                    rows = await conn.fetch('SELECT paper_name, figure_path, phash, dhash FROM figure_hashes')
            finally:
                await release_cloud_sql_conn(conn)

    if not rows:
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        if exclude_paper:
            cursor.execute(
                'SELECT paper_name, figure_path, phash, dhash FROM figure_hashes WHERE paper_name != ?',
                (exclude_paper,)
            )
        else:
            cursor.execute('SELECT paper_name, figure_path, phash, dhash FROM figure_hashes')
        rows = cursor.fetchall()
        conn.close()

    matches = []
    for paper_name, fig_path, stored_phash, stored_dhash in rows:
        p_dist = calculate_hamming_distance(query_phash, stored_phash)
        d_dist = calculate_hamming_distance(query_dhash, stored_dhash)
        avg_distance = (p_dist + d_dist) / 2.0

        if avg_distance <= threshold:
            similarity = max(0.0, (1.0 - (avg_distance / 64.0)) * 100.0)
            matches.append({
                "matched_paper": paper_name,
                "matched_figure": fig_path,
                "phash_distance": p_dist,
                "dhash_distance": d_dist,
                "hybrid_distance": avg_distance,
                "similarity_score": round(similarity, 2)
            })
    
    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    return matches


async def get_stats():
    if USE_CLOUD_SQL and INSTANCE_CONNECTION_NAME:
        conn = await get_cloud_sql_conn()
        if conn:
            try:
                total = await conn.fetchval('SELECT COUNT(*) FROM figure_hashes')
                papers = await conn.fetchval('SELECT COUNT(DISTINCT paper_name) FROM figure_hashes')
                return {"total_figures": total, "total_papers": papers}
            finally:
                await release_cloud_sql_conn(conn)

    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM figure_hashes')
    total = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(DISTINCT paper_name) FROM figure_hashes')
    papers = cursor.fetchone()[0]
    conn.close()
    return {"total_figures": total, "total_papers": papers}


async def close_pool():
    global _pool, _connector
    if _pool:
        await _pool.close()
        _pool = None
    if _connector:
        _connector.close()
        _connector = None