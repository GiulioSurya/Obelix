"""
Database Connection module - Gestione connessioni ai database.
"""

from src.connections.db_connection.abstract_db_connection import (
    AbstractDatabaseConnection,
    DatabaseConfig
)
from src.connections.db_connection.oracle_connection import (
    OracleConnection,
    OracleConfig,
    get_oracle_connection
)
from src.connections.db_connection.postgres_connection import (
    PostgresConnection,
    PostgresConfig,
    get_postgres_connection
)


__all__ = [
    # Abstract base classes
    "AbstractDatabaseConnection",
    "DatabaseConfig",
    # Concrete implementations
    "OracleConnection",
    "OracleConfig",
    "PostgresConnection",
    "PostgresConfig",
    # Helper functions (backward compatible)
    "get_oracle_connection",
    "get_postgres_connection",
]
