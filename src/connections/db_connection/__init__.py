"""
Database Connection module - Unified database connection management.

This module provides abstract interfaces and concrete implementations
for database connections, with configurable security and uniform results.
"""

from src.connections.db_connection.abstract_db_connection import (
    AbstractDatabaseConnection,
    DatabaseConfig
)
from src.connections.db_connection.query_result import (
    QueryResult,
    QueryStatus,
    ColumnMetadata
)
from src.connections.db_connection.security import (
    SecurityMode,
    SQLValidator,
    SQLSecurityException,
    ReadOnlySQLValidator,
    NoOpValidator,
    create_validator
)
from src.connections.db_connection.oracle_connection import (
    OracleConnection,
    OracleConfig,
    ConnectionMethod,
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

    # Query results
    "QueryResult",
    "QueryStatus",
    "ColumnMetadata",

    # Security
    "SecurityMode",
    "SQLValidator",
    "SQLSecurityException",
    "ReadOnlySQLValidator",
    "NoOpValidator",
    "create_validator",

    # Oracle
    "OracleConnection",
    "OracleConfig",
    "ConnectionMethod",
    "get_oracle_connection",

    # PostgreSQL
    "PostgresConnection",
    "PostgresConfig",
    "get_postgres_connection",
]
