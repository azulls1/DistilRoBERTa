"""Pool de conexiones a Postgres (Supabase maestría) con filas como diccionarios."""
from contextlib import contextmanager

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import config

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(config().database_url, min_size=1, max_size=5, open=True,
                               kwargs={"row_factory": dict_row, "autocommit": True})
    return _pool


@contextmanager
def cursor():
    with pool().connection() as conn, conn.cursor() as cur:
        yield cur


def todos(sql: str, params: tuple | dict = ()) -> list[dict]:
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def uno(sql: str, params: tuple | dict = ()) -> dict | None:
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def cerrar() -> None:
    if _pool is not None:
        _pool.close()
