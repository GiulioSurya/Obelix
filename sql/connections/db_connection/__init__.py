"""
Database Connection module - Gestione connessioni ai database.
"""

from sql.connections.db_connection.oracle_connection import (
    # Config
    OraclePoolConfig,
    # Pool factory (async)
    create_oracle_pool,
    close_oracle_pool,
    # Helpers async
    acquire_connection,
    execute_query,
    # Sync (per script offline)
    get_sync_connection,
    execute_query_sync,
    # Security
    SQLValidator,
    SQLSecurityException,
)
from sql.connections.db_connection.postgres_connection import (
    PostgresConnection,
    PostgresConfig,
    get_postgres_connection
)


__all__ = [
    # Oracle - Config
    "OraclePoolConfig",
    # Oracle - Pool factory (async)
    "create_oracle_pool",
    "close_oracle_pool",
    # Oracle - Helpers async
    "acquire_connection",
    "execute_query",
    # Oracle - Sync (per script offline)
    "get_sync_connection",
    "execute_query_sync",
    # Oracle - Security
    "SQLValidator",
    "SQLSecurityException",
    # Postgres
    "PostgresConnection",
    "PostgresConfig",
    "get_postgres_connection",
]
