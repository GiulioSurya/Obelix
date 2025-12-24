"""
Abstract base classes for database connections.

This module provides abstract interfaces for database connection management,
configuration, and query execution. All concrete database connection implementations
must inherit from these base classes.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, TypeVar, Generic


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

    Pattern:
        - Singleton: One instance per connection type
        - Thread-safe: Uses threading.Lock for initialization
        - Configurable: Accepts config object or loads from K8sConfig
    """

    _instance: Optional['AbstractDatabaseConnection'] = None
    _lock: threading.Lock = threading.Lock()
    _config: Optional[TConfig] = None

    # ============================================================================
    # CORE METHODS - Must be implemented by all subclasses
    # ============================================================================

    @classmethod
    @abstractmethod
    def get_instance(cls, config: Optional[TConfig] = None) -> 'AbstractDatabaseConnection':
        """
        Get singleton instance of the database connection.

        Thread-safe singleton implementation. If config is provided or instance
        is not yet configured, (re)initializes with the provided config.

        Args:
            config: Database configuration (optional). If None, loads from K8sConfig.

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

    # ============================================================================
    # QUERY EXECUTION - Must be implemented by all subclasses
    # ============================================================================

    @abstractmethod
    def execute_query(
        self,
        query: str,
        params: Optional[Union[Dict[str, Any], Tuple[Any, ...]]] = None,
        **kwargs
    ) -> Any:
        """
        Execute SQL query and return results.

        Parameter style varies by database:
        - Oracle: Dict with named parameters {:param_name}
        - Postgres: Tuple with positional parameters %s

        Args:
            query: SQL query string
            params: Query parameters (Dict for Oracle, Tuple for Postgres)
            **kwargs: Additional database-specific parameters

        Returns:
            Query results (format depends on implementation)
        """
        pass

    # ============================================================================
    # METADATA - Must be implemented by all subclasses
    # ============================================================================

    @abstractmethod
    def get_database_version(self) -> Dict[str, Any]:
        """
        Get database version and connection information.

        Returns:
            Dict: Database-specific version information
                - version/version_banner: Database version string
                - database/database_name: Database name
                - user/current_user: Connected user
                - connection_status: Connection status
        """
        pass

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

    def execute_query_dict(
        self,
        query: str,
        params: Optional[Union[Dict[str, Any], Tuple[Any, ...]]] = None
    ) -> Any:
        """
        Execute query and return results as list of dictionaries.

        Default: Not implemented (raises NotImplementedError)
        Override: PostgresConnection

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List[Dict[str, Any]]: Query results as dictionaries

        Raises:
            NotImplementedError: If not supported by database implementation
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support execute_query_dict()"
        )