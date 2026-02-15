"""
PostgreSQL Database Connection Singleton
=========================================

Thread-safe singleton class for managing PostgreSQL database connections using psycopg.
Supports connection pooling and provides methods for common operations including
table creation, data insertion, and query execution.

Requirements:
    pip install psycopg[binary,pool]

Environment Variables:
    POSTGRES_HOST: Database host
    POSTGRES_PORT: Database port (default: 5432)
    POSTGRES_DB: Database name
    POSTGRES_USER: Username
    POSTGRES_PASSWORD: Password
    POSTGRES_EMBED_DIM: Embedding dimension for vector operations (default: 1024)
"""

import os
import threading
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError:
    raise ImportError(
        "psycopg package not found. Install with: pip install psycopg[binary,pool]"
    )


@dataclass
class PostgresConfig:
    """PostgreSQL database connection configuration"""
    host: str
    port: int
    database: str
    user: str
    password: str
    embed_dim: int = 1024

    @classmethod
    def from_k8s_config(cls) -> 'PostgresConfig':
        """
        Crea configurazione da K8sConfig (infrastructure.yaml).

        Tutte le configurazioni (host, port, database, user, password, embed_dim)
        vengono lette da infrastructure.yaml.

        Returns:
            PostgresConfig configurato

        Raises:
            ValueError: Se mancano credenziali o configurazione database
        """
        from obelix.infrastructure.k8s import YamlConfig
        import os

        # Leggi tutte le configurazioni da infrastructure.yaml (include credenziali)
        infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
        postgres_db_config = infra_config.get("databases.postgres")

        return cls(
            host=postgres_db_config["host"],
            port=postgres_db_config["port"],
            database=postgres_db_config["database"],
            user=postgres_db_config["user"],
            password=postgres_db_config["password"],
            embed_dim=postgres_db_config["embed_dim"]
        )

    def build_connection_string(self) -> str:
        """Build PostgreSQL connection string"""
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password}"
        )


class PostgresConnection:
    """
    Thread-safe singleton for PostgreSQL database connections with connection pooling.

    Example usage:
        # Using environment variables
        conn = PostgresConnection.get_instance()

        # Using custom configuration
        config = PostgresConfig(
            host="localhost",
            port=5432,
            database="mydb",
            user="myuser",
            password="mypass"
        )
        conn = PostgresConnection.get_instance(config)

        # Execute query with context manager
        with conn.get_connection() as db_conn:
            with db_conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE id = %s", (1,))
                results = cursor.fetchall()

        # Or use helper methods
        results = conn.execute_query("SELECT * FROM users WHERE id = %s", (1,))

        # Create table
        conn.create_table(
            "users",
            {
                "id": "SERIAL PRIMARY KEY",
                "name": "VARCHAR(100) NOT NULL",
                "email": "VARCHAR(255) UNIQUE"
            }
        )

        # Insert data
        conn.insert_data("users", {"name": "John", "email": "john@example.com"})
    """

    _instance: Optional['PostgresConnection'] = None
    _lock: threading.Lock = threading.Lock()
    _pool: Optional[ConnectionPool] = None
    _config: Optional[PostgresConfig] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls, config: Optional[PostgresConfig] = None) -> 'PostgresConnection':
        """
        Get singleton instance with optional configuration.

        Args:
            config: PostgreSQL configuration. If None, will try to load from environment.

        Returns:
            PostgresConnection instance
        """
        instance = cls()

        # Set configuration if provided or if not already set
        if config is not None or cls._config is None:
            if config is None:
                config = PostgresConfig.from_k8s_config()
            cls._config = config
            # Reset pool if config changes
            if cls._pool is not None:
                cls._pool.close()
                cls._pool = None

        return instance

    @staticmethod
    def _check_connection(conn: psycopg.Connection) -> None:
        """Health check eseguito al checkout dal pool."""
        conn.execute("SELECT 1")

    def _create_pool(self) -> ConnectionPool:
        """Create a new connection pool"""
        if self._config is None:
            raise RuntimeError("No configuration provided")

        try:
            pool = ConnectionPool(
                conninfo=self._config.build_connection_string(),
                min_size=2,
                max_size=10,
                timeout=30,
                max_idle=60,
                max_lifetime=3600,
                check=self._check_connection,
            )
            return pool
        except Exception as e:
            raise RuntimeError(f"Failed to create PostgreSQL connection pool: {e}")

    def get_pool(self) -> ConnectionPool:
        """
        Get connection pool. Creates new pool if needed.

        Returns:
            PostgreSQL connection pool
        """
        with self._lock:
            if self._pool is None:
                self._pool = self._create_pool()
            return self._pool

    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a connection from the pool.

        Yields:
            PostgreSQL connection from pool
        """
        pool = self.get_pool()
        with pool.connection() as conn:
            yield conn

    def close_pool(self):
        """Close the connection pool"""
        with self._lock:
            if self._pool is not None:
                try:
                    self._pool.close()
                except Exception:
                    pass
                finally:
                    self._pool = None

    def close_connection(self):
        """Alias for close_pool() (compatibilità interna)."""
        self.close_pool()

    # =============================================================================
    # Async wrappers (per uso in FastAPI / asyncio)
    # =============================================================================

    async def execute_query_async(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
        fetch_results: bool = True
    ) -> List[Tuple[Any, ...]]:
        """
        Async wrapper per execute_query() che evita di bloccare l'event loop.
        """
        return await asyncio.to_thread(self.execute_query, query, params, fetch_results)

    async def execute_query_dict_async(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None
    ) -> List[Dict[str, Any]]:
        """
        Async wrapper per execute_query_dict() che evita di bloccare l'event loop.
        """
        return await asyncio.to_thread(self.execute_query_dict, query, params)

    async def insert_data_async(
        self,
        table_name: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Optional[Any]:
        """
        Async wrapper per insert_data() che evita di bloccare l'event loop.
        """
        return await asyncio.to_thread(self.insert_data, table_name, data, **kwargs)

    def execute_query(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
        fetch_results: bool = True
    ) -> List[Tuple[Any, ...]]:
        """
        Execute a query and optionally return results.

        Args:
            query: SQL query string (use %s for parameters)
            params: Query parameters as tuple
            fetch_results: If True, fetch and return results (for SELECT queries)

        Returns:
            List of result tuples for SELECT queries, empty list otherwise
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)

                if fetch_results:
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return []

    def execute_query_dict(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a query and return results as list of dictionaries.

        Args:
            query: SQL query string (use %s for parameters)
            params: Query parameters as tuple

        Returns:
            List of result dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()

    def create_table(
        self,
        table_name: str,
        columns: Dict[str, str],
        if_not_exists: bool = True
    ) -> None:
        """
        Create a table with specified columns.

        Args:
            table_name: Name of the table to create
            columns: Dictionary mapping column names to their SQL type definitions
            if_not_exists: If True, use IF NOT EXISTS clause

        Example:
            create_table(
                "users",
                {
                    "id": "SERIAL PRIMARY KEY",
                    "name": "VARCHAR(100) NOT NULL",
                    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                }
            )
        """
        if_not_exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        columns_def = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
        query = f"CREATE TABLE {if_not_exists_clause}{table_name} ({columns_def})"

        self.execute_query(query, fetch_results=False)

    def insert_data(
        self,
        table_name: str,
        data: Dict[str, Any],
        returning: Optional[str] = None
    ) -> Optional[Any]:
        """
        Insert a single row into a table.

        Args:
            table_name: Name of the table
            data: Dictionary mapping column names to values
            returning: Optional column name to return (e.g., "id")

        Returns:
            Value of the returning column if specified, None otherwise

        Example:
            user_id = insert_data(
                "users",
                {"name": "John", "email": "john@example.com"},
                returning="id"
            )
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        values = tuple(data.values())

        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        if returning:
            query += f" RETURNING {returning}"

        if returning:
            results = self.execute_query(query, values, fetch_results=True)
            return results[0][0] if results else None
        else:
            self.execute_query(query, values, fetch_results=False)
            return None

    def insert_many(
        self,
        table_name: str,
        data: List[Dict[str, Any]]
    ) -> None:
        """
        Insert multiple rows into a table efficiently.

        Args:
            table_name: Name of the table
            data: List of dictionaries mapping column names to values

        Example:
            insert_many(
                "users",
                [
                    {"name": "John", "email": "john@example.com"},
                    {"name": "Jane", "email": "jane@example.com"}
                ]
            )
        """
        if not data:
            return

        columns = ", ".join(data[0].keys())
        placeholders = ", ".join(["%s"] * len(data[0]))
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        values_list = [tuple(row.values()) for row in data]

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(query, values_list)
                conn.commit()

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise
        """
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = %s
            )
        """
        result = self.execute_query(query, (table_name,))
        return result[0][0] if result else False

    def get_database_version(self) -> Dict[str, Any]:
        """
        Get PostgreSQL database version information.

        Returns:
            Dictionary with version details
        """
        version_info = {}

        try:
            # Get PostgreSQL version
            result = self.execute_query("SELECT version()")
            if result:
                version_info['version'] = result[0][0]

            # Get current database name
            result = self.execute_query("SELECT current_database()")
            if result:
                version_info['database'] = result[0][0]

            # Get current user
            result = self.execute_query("SELECT current_user")
            if result:
                version_info['user'] = result[0][0]

            version_info['connection_status'] = 'Connected successfully'

        except Exception as e:
            version_info['error'] = str(e)
            version_info['connection_status'] = 'Connection failed'

        return version_info

    # ========================================================================
    # DDL (Schema Management) Methods
    # ========================================================================

    def drop_table(
        self,
        table_name: str,
        if_exists: bool = True,
        cascade: bool = False
    ) -> None:
        """
        Drop a table from the database.

        Args:
            table_name: Name of the table to drop
            if_exists: If True, use IF EXISTS clause (no error if table doesn't exist)
            cascade: If True, automatically drop dependent objects (views, constraints, etc.)

        Example:
            drop_table("old_users")
            drop_table("temp_data", cascade=True)
        """
        if_exists_clause = "IF EXISTS " if if_exists else ""
        cascade_clause = " CASCADE" if cascade else ""
        query = f"DROP TABLE {if_exists_clause}{table_name}{cascade_clause}"
        self.execute_query(query, fetch_results=False)

    def add_column(
        self,
        table_name: str,
        column_name: str,
        column_type: str,
        if_not_exists: bool = True
    ) -> None:
        """
        Add a column to an existing table.

        Args:
            table_name: Name of the table
            column_name: Name of the column to add
            column_type: SQL type definition (e.g., "VARCHAR(100) NOT NULL", "INTEGER DEFAULT 0")
            if_not_exists: If True, use IF NOT EXISTS clause (Postgres 9.6+)

        Example:
            add_column("users", "age", "INTEGER")
            add_column("users", "email", "VARCHAR(255) UNIQUE NOT NULL")
        """
        if_not_exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        query = f"ALTER TABLE {table_name} ADD COLUMN {if_not_exists_clause}{column_name} {column_type}"
        self.execute_query(query, fetch_results=False)

    def drop_column(
        self,
        table_name: str,
        column_name: str,
        if_exists: bool = True,
        cascade: bool = False
    ) -> None:
        """
        Drop a column from a table.

        Args:
            table_name: Name of the table
            column_name: Name of the column to drop
            if_exists: If True, use IF EXISTS clause
            cascade: If True, automatically drop dependent objects

        Example:
            drop_column("users", "deprecated_field")
            drop_column("users", "old_data", cascade=True)
        """
        if_exists_clause = "IF EXISTS " if if_exists else ""
        cascade_clause = " CASCADE" if cascade else ""
        query = f"ALTER TABLE {table_name} DROP COLUMN {if_exists_clause}{column_name}{cascade_clause}"
        self.execute_query(query, fetch_results=False)

    def create_index(
        self,
        index_name: str,
        table_name: str,
        columns: str,
        method: Optional[str] = None,
        if_not_exists: bool = True,
        unique: bool = False
    ) -> None:
        """
        Create an index on a table.

        Args:
            index_name: Name of the index
            table_name: Name of the table
            columns: Column expression (e.g., "column_name" or "embedding vector_cosine_ops")
            method: Index method (e.g., "HNSW", "IVFFlat", "BTREE", "HASH")
            if_not_exists: If True, check existence before creating (using custom logic)
            unique: If True, create a UNIQUE index

        Example:
            # Regular B-tree index
            create_index("idx_user_email", "users", "email")

            # Vector similarity index (pgvector with HNSW)
            create_index("idx_embeddings", "docs", "embedding vector_cosine_ops", method="HNSW")

            # Unique index
            create_index("idx_user_username", "users", "username", unique=True)
        """
        # Postgres doesn't support IF NOT EXISTS for CREATE INDEX directly
        # We need to check manually
        if if_not_exists and self.index_exists(index_name):
            return

        unique_clause = "UNIQUE " if unique else ""
        using_clause = f" USING {method}" if method else ""
        query = f"CREATE {unique_clause}INDEX {index_name} ON {table_name}{using_clause} ({columns})"
        self.execute_query(query, fetch_results=False)

    def drop_index(
        self,
        index_name: str,
        if_exists: bool = True,
        cascade: bool = False
    ) -> None:
        """
        Drop an index from the database.

        Args:
            index_name: Name of the index to drop
            if_exists: If True, use IF EXISTS clause
            cascade: If True, automatically drop dependent objects

        Example:
            drop_index("idx_old_column")
            drop_index("idx_deprecated", cascade=True)
        """
        if_exists_clause = "IF EXISTS " if if_exists else ""
        cascade_clause = " CASCADE" if cascade else ""
        query = f"DROP INDEX {if_exists_clause}{index_name}{cascade_clause}"
        self.execute_query(query, fetch_results=False)

    def truncate_table(
        self,
        table_name: str,
        restart_identity: bool = False,
        cascade: bool = False
    ) -> None:
        """
        Truncate a table (remove all rows, faster than DELETE).

        Args:
            table_name: Name of the table to truncate
            restart_identity: If True, restart sequences (e.g., SERIAL columns back to 1)
            cascade: If True, automatically truncate dependent tables

        Example:
            truncate_table("temp_logs")
            truncate_table("sessions", restart_identity=True)
        """
        restart_clause = " RESTART IDENTITY" if restart_identity else ""
        cascade_clause = " CASCADE" if cascade else ""
        query = f"TRUNCATE TABLE {table_name}{restart_clause}{cascade_clause}"
        self.execute_query(query, fetch_results=False)

    def enable_extension(self, extension_name: str) -> None:
        """
        Enable a PostgreSQL extension.

        Args:
            extension_name: Name of the extension (e.g., "vector" for pgvector)

        Example:
            enable_extension("vector")  # Enable pgvector for vector operations
            enable_extension("uuid-ossp")  # Enable UUID generation
        """
        query = f"CREATE EXTENSION IF NOT EXISTS {extension_name}"
        self.execute_query(query, fetch_results=False)

    def update_data(
        self,
        table_name: str,
        data: Dict[str, Any],
        where_clause: str,
        where_params: Optional[Tuple[Any, ...]] = None
    ) -> int:
        """
        Update rows in a table.

        Args:
            table_name: Name of the table
            data: Dictionary mapping column names to new values
            where_clause: WHERE clause (without WHERE keyword, use %s for parameters)
            where_params: Parameters for WHERE clause placeholders

        Returns:
            Number of rows updated

        Example:
            # Update single row
            count = update_data("users", {"name": "John Doe"}, "id = %s", (1,))

            # Update multiple rows
            count = update_data("users", {"active": False}, "last_login < %s", (cutoff_date,))
        """
        set_clause = ", ".join([f"{col} = %s" for col in data.keys()])
        values = list(data.values())

        if where_params:
            values.extend(where_params)

        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, tuple(values))
                row_count = cursor.rowcount
                conn.commit()
                return row_count

    def delete_data(
        self,
        table_name: str,
        where_clause: str,
        where_params: Optional[Tuple[Any, ...]] = None
    ) -> int:
        """
        Delete rows from a table.

        Args:
            table_name: Name of the table
            where_clause: WHERE clause (without WHERE keyword, use %s for parameters)
            where_params: Parameters for WHERE clause placeholders

        Returns:
            Number of rows deleted

        Example:
            # Delete single row
            count = delete_data("users", "id = %s", (1,))

            # Delete multiple rows
            count = delete_data("logs", "created_at < %s", (cutoff_date,))
        """
        query = f"DELETE FROM {table_name} WHERE {where_clause}"

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, where_params)
                row_count = cursor.rowcount
                conn.commit()
                return row_count

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """
        Check if a column exists in a table.

        Args:
            table_name: Name of the table
            column_name: Name of the column

        Returns:
            True if column exists, False otherwise

        Example:
            if column_exists("users", "email"):
                print("Email column exists")
        """
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = %s
                AND column_name = %s
            )
        """
        result = self.execute_query(query, (table_name, column_name))
        return result[0][0] if result else False

    def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists in the database.

        Args:
            index_name: Name of the index

        Returns:
            True if index exists, False otherwise

        Example:
            if not index_exists("idx_user_email"):
                create_index("idx_user_email", "users", "email")
        """
        query = """
            SELECT EXISTS (
                SELECT FROM pg_indexes
                WHERE schemaname = 'public'
                AND indexname = %s
            )
        """
        result = self.execute_query(query, (index_name,))
        return result[0][0] if result else False


# Convenience function for quick access
def get_postgres_connection(config: Optional[PostgresConfig] = None) -> PostgresConnection:
    """
    Convenience function to get PostgreSQL connection singleton.

    Args:
        config: Optional PostgreSQL configuration

    Returns:
        PostgresConnection instance
    """
    return PostgresConnection.get_instance(config)


if __name__ == "__main__":
    # Example usage - test connection and list tables
    try:
        # Test connection using environment variables
        pg_conn = get_postgres_connection()

        # Test connection and get database info
        print("=== PostgreSQL Connection Test ===")
        version_info = pg_conn.get_database_version()
        for key, value in version_info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        # List all tables in the database
        print("\n=== Listing All Tables in Database ===")
        query = """
            SELECT
                table_schema,
                table_name,
                table_type
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
        """
        tables = pg_conn.execute_query_dict(query)

        if tables:
            print(f"\nFound {len(tables)} tables/views:")
            for table in tables:
                schema = table.get('table_schema', 'N/A')
                name = table.get('table_name', 'N/A')
                type_ = table.get('table_type', 'N/A')
                print(f"  - {schema}.{name} ({type_})")
        else:
            print("No tables found in the database.")

        # Count rows in each table (if any tables exist)
        if tables:
            print("\n=== Row Counts for Each Table ===")
            for table in tables:
                schema = table.get('table_schema')
                name = table.get('table_name')
                try:
                    count_query = f'SELECT COUNT(*) FROM "{schema}"."{name}"'
                    result = pg_conn.execute_query(count_query)
                    count = result[0][0] if result else 0
                    print(f"  - {schema}.{name}: {count} rows")
                except Exception as e:
                    print(f"  - {schema}.{name}: Error counting rows ({str(e)[:50]})")

    except Exception as e:
        print(f"Connection test failed: {e}")
        print("\nMake sure to set environment variables:")
        print("- POSTGRES_HOST")
        print("- POSTGRES_PORT (optional, default: 5432)")
        print("- POSTGRES_DB")
        print("- POSTGRES_USER")
        print("- POSTGRES_PASSWORD")
