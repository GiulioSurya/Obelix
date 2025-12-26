"""
Abstract base classes for database connections.

This module provides abstract interfaces for database connection management,
configuration, and query execution. All concrete database connection implementations
must inherit from these base classes.
"""

import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic

from src.connections.db_connection.query_result import (
    QueryResult,
    ColumnMetadata,
)
from src.connections.db_connection.security import (
    SecurityMode,
    SQLValidator,
    create_validator,
)


# Type variables for generic connection and config types
TConfig = TypeVar('TConfig', bound='DatabaseConfig')
TConnection = TypeVar('TConnection')


@dataclass
class DatabaseConfig(ABC):
    """
    Abstract base class for database configuration.

    Each database type must provide its own configuration class
    that implements connection string building and K8s config loading.
    """

    @classmethod
    @abstractmethod
    def from_k8s_config(cls) -> 'DatabaseConfig':
        """
        Create configuration from K8sConfig and environment variables.

        Returns:
            DatabaseConfig: Configured database connection settings
        """
        pass

    @abstractmethod
    def build_connection_string(self) -> str:
        """
        Build database-specific connection string.

        Returns:
            str: Connection string formatted for the specific database driver
        """
        pass


class AbstractDatabaseConnection(ABC, Generic[TConfig, TConnection]):
    """
    Abstract base class for database connections with Singleton pattern.

    This class provides the interface for all database connection managers.
    Each implementation handles its specific database type (Oracle, Postgres, etc.)

    Type Parameters:
        TConfig: Configuration class type (e.g., OracleConfig, PostgresConfig)
        TConnection: Connection object type (e.g., oracledb.Connection, psycopg.Connection)

    Features:
        - Singleton: One instance per connection type
        - Thread-safe: Uses threading.Lock for initialization
        - Configurable security: SecurityMode controls allowed operations
        - Uniform results: execute_query returns QueryResult
    """

    _instance: Optional['AbstractDatabaseConnection'] = None
    _lock: threading.Lock = threading.Lock()
    _config: Optional[TConfig] = None

    def __init__(
        self,
        security_mode: SecurityMode = SecurityMode.READ_ONLY,
        custom_validator: Optional[SQLValidator] = None
    ):
        """
        Initialize connection with security settings.

        Args:
            security_mode: Security mode (default: READ_ONLY for AI safety)
            custom_validator: Custom validator (required if mode is CUSTOM)
        """
        self._security_mode = security_mode
        self._validator = create_validator(security_mode, custom_validator)

    # ============================================================================
    # ABSTRACT METHODS - Must be implemented by all subclasses
    # ============================================================================

    @classmethod
    @abstractmethod
    def get_instance(
        cls,
        config: Optional[TConfig] = None,
        security_mode: SecurityMode = SecurityMode.READ_ONLY,
        **kwargs
    ) -> 'AbstractDatabaseConnection':
        """
        Get singleton instance of the database connection.

        Thread-safe singleton implementation. If config is provided or instance
        is not yet configured, (re)initializes with the provided config.

        Args:
            config: Database configuration (optional). If None, loads from K8sConfig.
            security_mode: Security mode for SQL validation.

        Returns:
            AbstractDatabaseConnection: Singleton instance
        """
        pass

    @abstractmethod
    def get_connection(self) -> Union[TConnection, Any]:
        """
        Get database connection.

        Implementation varies by database:
        - OracleConnection: Returns direct connection object
        - PostgresConnection: Returns context manager for connection pool

        Returns:
            Connection object or context manager, depending on implementation
        """
        pass

    @abstractmethod
    def close_connection(self) -> None:
        """
        Close database connection or connection pool.

        Implementation varies by database:
        - OracleConnection: Closes single connection
        - PostgresConnection: Closes connection pool
        """
        pass

    @abstractmethod
    def _execute_raw(
        self,
        query: str,
        params: Optional[Union[Dict[str, Any], Tuple[Any, ...]]] = None
    ) -> Tuple[List[Tuple[Any, ...]], List[ColumnMetadata]]:
        """
        Execute query without validation (internal use).

        This is the database-specific execution logic. Validation is handled
        by execute_query() which calls this method.

        Args:
            query: SQL query string
            params: Query parameters (Dict for Oracle :named, Tuple for Postgres %s)

        Returns:
            Tuple of (rows, column_metadata)
        """
        pass

    @abstractmethod
    def get_database_version(self) -> Dict[str, Any]:
        """
        Get database version and connection information.

        Returns:
            Dict: Database-specific version information
        """
        pass

    @property
    @abstractmethod
    def param_style(self) -> str:
        """
        Parameter style for this database.

        Returns:
            'named' for Oracle (:param), 'positional' for PostgreSQL (%s)
        """
        pass

    # ============================================================================
    # CONCRETE METHODS - Query execution with validation
    # ============================================================================

    def execute_query(
        self,
        query: str,
        params: Optional[Union[Dict[str, Any], Tuple[Any, ...]]] = None
    ) -> QueryResult:
        """
        Execute SQL query with validation and return uniform QueryResult.

        Security validation is applied based on the connection's SecurityMode.

        Args:
            query: SQL query string
            params: Query parameters (Dict for Oracle :named, Tuple for Postgres %s)

        Returns:
            QueryResult with rows, columns, status
        """
        start_time = time.perf_counter()

        try:
            # Validate query based on security mode
            self._validator.validate(query)

            # Execute raw query
            rows, columns = self._execute_raw(query, params)

            execution_time = (time.perf_counter() - start_time) * 1000
            return QueryResult.success(rows, columns, execution_time_ms=execution_time)

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return QueryResult.error(str(e), execution_time_ms=execution_time)

    def execute_query_dict(
        self,
        query: str,
        params: Optional[Union[Dict[str, Any], Tuple[Any, ...]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute query and return results as list of dictionaries.

        Convenience method that calls execute_query() and converts to dicts.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List[Dict[str, Any]]: Query results as dictionaries

        Raises:
            Exception: If query fails
        """
        result = self.execute_query(query, params)
        if not result.is_success:
            raise Exception(result.error)
        return result.as_dicts()

    # ============================================================================
    # TRANSACTION MANAGEMENT
    # ============================================================================

    @contextmanager
    def transaction(self):
        """
        Context manager for transactions.

        Usage:
            with db.transaction() as conn:
                db.execute_query("INSERT ...")
                db.execute_query("UPDATE ...")
            # Auto-commit on success, rollback on exception
        """
        conn = self.get_connection()
        try:
            yield conn
            self._commit(conn)
        except Exception:
            self._rollback(conn)
            raise

    def _commit(self, conn: TConnection) -> None:
        """Commit transaction (override if needed)"""
        if hasattr(conn, 'commit'):
            conn.commit()

    def _rollback(self, conn: TConnection) -> None:
        """Rollback transaction (override if needed)"""
        if hasattr(conn, 'rollback'):
            conn.rollback()

    # ============================================================================
    # OPTIONAL METHODS - Override only if supported by database
    # ============================================================================

    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists in database.

        Default: Not implemented (raises NotImplementedError)
        Override: PostgresConnection

        Args:
            table_name: Name of the table to check

        Returns:
            bool: True if table exists

        Raises:
            NotImplementedError: If not supported by database implementation
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support table_exists()"
        )

    def create_table(
        self,
        table_name: str,
        columns: Dict[str, str],
        **kwargs
    ) -> None:
        """
        Create table with specified columns.

        Default: Not implemented (raises NotImplementedError)
        Override: PostgresConnection

        Args:
            table_name: Name of the table to create
            columns: Dict mapping column names to SQL types
            **kwargs: Additional database-specific options

        Raises:
            NotImplementedError: If not supported by database implementation
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support create_table()"
        )

    def drop_table(
        self,
        table_name: str,
        **kwargs
    ) -> None:
        """
        Drop table from database.

        Default: Not implemented (raises NotImplementedError)
        Override: PostgresConnection

        Args:
            table_name: Name of the table to drop
            **kwargs: Additional database-specific options

        Raises:
            NotImplementedError: If not supported by database implementation
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support drop_table()"
        )

    def insert_data(
        self,
        table_name: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Optional[Any]:
        """
        Insert data into table.

        Default: Not implemented (raises NotImplementedError)
        Override: PostgresConnection

        Args:
            table_name: Name of the table
            data: Dict mapping column names to values
            **kwargs: Additional database-specific options

        Returns:
            Optional[Any]: Inserted row ID or None

        Raises:
            NotImplementedError: If not supported by database implementation
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support insert_data()"
        )
