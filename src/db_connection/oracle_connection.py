"""
Oracle Database Connection Singleton
=====================================

A simple singleton class for managing Oracle database connections using python-oracledb.
Supports multiple connection methods: Easy Connect, TNS, and detailed parameters.

Requirements:
    pip install oracledb

Environment Variables:
    ORACLE_HOST: Database host
    ORACLE_PORT: Database port (default: 1521)
    ORACLE_SERVICE_NAME: Service name
    ORACLE_USER: Username
    ORACLE_PASSWORD: Password
    ORACLE_DSN: Complete connection string (alternative to host/port/service)
"""

import os
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
load_dotenv()


try:
    import oracledb
    # Try to initialize thick mode for older Oracle versions
    try:
        # Try common Oracle client locations
        oracle_client_paths = [
            r"C:\Oracle\instantclient_23_9",   # Your installation
            r"C:\Oracle\instantclient_21_3",
            r"C:\Oracle\instantclient",
            r"C:\app\oracle\product\19.0.0\client_1\bin",
            r"C:\oracle\product\19.0.0\client_1\bin"
        ]

        thick_mode_initialized = False
        for path in oracle_client_paths:
            try:
                oracledb.init_oracle_client(lib_dir=path)
                print(f"Oracle thick mode initialized successfully with: {path}")
                thick_mode_initialized = True
                break
            except:
                continue

        if not thick_mode_initialized:
            # Try without specifying path
            oracledb.init_oracle_client()
            print("Oracle thick mode initialized successfully")

    except Exception as e:
        print(f"Thick mode initialization failed: {e}")
        print("Continuing with thin mode (requires Oracle 12.1+)")
        print("For older Oracle versions, install Oracle Instant Client:")
        print("https://www.oracle.com/database/technologies/instant-client/downloads.html")
except ImportError:
    raise ImportError("oracledb package not found. Install with: pip install oracledb")


class ConnectionMethod(Enum):
    """Supported Oracle connection methods"""
    EASY_CONNECT = "easy_connect"  # host:port/service_name
    DSN = "dsn"                   # Complete connection string
    TNS_ALIAS = "tns_alias"       # TNS alias from tnsnames.ora


@dataclass
class OracleConfig:
    """Oracle database connection configuration"""
    user: str
    password: str
    host: Optional[str] = None
    port: int = 1521
    service_name: Optional[str] = None
    dsn: Optional[str] = None
    method: ConnectionMethod = ConnectionMethod.EASY_CONNECT

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.method == ConnectionMethod.EASY_CONNECT:
            if not self.host or not self.service_name:
                raise ValueError("host and service_name are required for Easy Connect method")
        elif self.method == ConnectionMethod.DSN:
            if not self.dsn:
                raise ValueError("dsn is required for DSN method")

    @classmethod
    def from_env(cls) -> 'OracleConfig':
        """Create configuration from environment variables"""
        user = os.getenv('ORACLE_USER')
        password = os.getenv('ORACLE_PASSWORD')

        if not user or not password:
            raise ValueError("ORACLE_USER and ORACLE_PASSWORD environment variables are required")

        # Check if DSN is provided
        dsn = os.getenv('ORACLE_DSN')
        if dsn:
            return cls(
                user=user,
                password=password,
                dsn=dsn,
                method=ConnectionMethod.DSN
            )

        # Use Easy Connect method
        host = os.getenv('ORACLE_HOST')
        service_name = os.getenv('ORACLE_SERVICE_NAME')
        port = int(os.getenv('ORACLE_PORT', '1521'))

        return cls(
            user=user,
            password=password,
            host=host,
            port=port,
            service_name=service_name,
            method=ConnectionMethod.EASY_CONNECT
        )

    def build_connection_string(self) -> str:
        """Build connection string based on method"""
        if self.method == ConnectionMethod.DSN:
            return self.dsn
        elif self.method == ConnectionMethod.EASY_CONNECT:
            return f"{self.host}:{self.port}/{self.service_name}"
        elif self.method == ConnectionMethod.TNS_ALIAS:
            return self.dsn  # TNS alias name
        else:
            raise ValueError(f"Unsupported connection method: {self.method}")


class OracleConnectionSingleton:
    """
    Thread-safe singleton for Oracle database connections.

    Example usage:
        # Using environment variables
        conn = OracleConnectionSingleton.get_instance()

        # Using custom configuration
        config = OracleConfig(
            user="myuser",
            password="mypass",
            host="localhost",
            service_name="ORCLPDB"
        )
        conn = OracleConnectionSingleton.get_instance(config)

        # Get database connection
        with conn.get_connection() as cursor:
            cursor.execute("SELECT * FROM dual")
            result = cursor.fetchone()
    """

    _instance = None
    _lock = threading.Lock()
    _connection = None
    _config: Optional[OracleConfig] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls, config: Optional[OracleConfig] = None) -> 'OracleConnectionSingleton':
        """
        Get singleton instance with optional configuration.

        Args:
            config: Oracle configuration. If None, will try to load from environment.

        Returns:
            OracleConnectionSingleton instance
        """
        instance = cls()

        # Set configuration if provided or if not already set
        if config is not None or cls._config is None:
            if config is None:
                config = OracleConfig.from_env()
            cls._config = config
            # Reset connection if config changes
            if cls._connection is not None:
                try:
                    cls._connection.close()
                except:
                    pass
                cls._connection = None

        return instance

    def _create_connection(self):
        """Create a new Oracle database connection"""
        if self._config is None:
            raise RuntimeError("No configuration provided")

        connection_params = {
            'user': self._config.user,
            'password': self._config.password,
            'dsn': self._config.build_connection_string()
        }

        try:
            connection = oracledb.connect(**connection_params)
            return connection
        except oracledb.Error as e:
            raise RuntimeError(f"Failed to connect to Oracle database: {e}")

    def get_connection(self):
        """
        Get database connection. Creates new connection if needed.

        Returns:
            Oracle database connection
        """
        with self._lock:
            if self._connection is None or not self._is_connection_valid():
                self._connection = self._create_connection()
            return self._connection

    def _is_connection_valid(self) -> bool:
        """Check if current connection is valid"""
        if self._connection is None:
            return False

        try:
            # Simple ping to check connection
            cursor = self._connection.cursor()
            cursor.execute("SELECT 1 FROM dual")
            cursor.close()
            return True
        except:
            return False

    def close_connection(self):
        """Close the database connection"""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except:
                    pass
                finally:
                    self._connection = None

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> list:
        """
        Execute a SELECT query and return results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of query results
        """
        connection = self.get_connection()
        cursor = connection.cursor()

        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            return cursor.fetchall()
        finally:
            cursor.close()

    def get_database_version(self) -> dict:
        """
        Get Oracle database version information using available methods.
        Works with limited privileges.

        Returns:
            Dictionary with version details
        """
        connection = self.get_connection()
        cursor = connection.cursor()

        try:
            version_info = {}

            # Try to get version info with different methods
            methods_tried = []

            # Method 1: Try v$version (requires privileges)
            try:
                cursor.execute("SELECT banner FROM v$version WHERE banner LIKE 'Oracle Database%'")
                version_result = cursor.fetchone()
                if version_result:
                    version_info['version_banner'] = version_result[0]
                    methods_tried.append("v$version")
            except Exception as e:
                methods_tried.append(f"v$version (failed: {str(e)[:50]}...)")

            # Method 2: Try v$instance (requires privileges)
            try:
                cursor.execute("SELECT version FROM v$instance")
                version_number = cursor.fetchone()
                if version_number:
                    version_info['version_number'] = version_number[0]
                    methods_tried.append("v$instance")
            except Exception as e:
                methods_tried.append(f"v$instance (failed: {str(e)[:50]}...)")

            # Method 3: Try getting version from connection itself
            try:
                db_version = connection.version
                version_info['connection_version'] = db_version
                methods_tried.append("connection.version")
            except Exception as e:
                methods_tried.append(f"connection.version (failed: {str(e)[:50]}...)")

            # Method 4: Try SYS_CONTEXT (usually works for any user)
            try:
                cursor.execute("SELECT SYS_CONTEXT('USERENV', 'DB_NAME') FROM dual")
                db_name = cursor.fetchone()
                if db_name:
                    version_info['database_name'] = db_name[0]
                    methods_tried.append("SYS_CONTEXT DB_NAME")
            except Exception as e:
                methods_tried.append(f"SYS_CONTEXT (failed: {str(e)[:50]}...)")

            # Method 5: Try getting user info
            try:
                cursor.execute("SELECT USER FROM dual")
                current_user = cursor.fetchone()
                if current_user:
                    version_info['current_user'] = current_user[0]
                    methods_tried.append("USER info")
            except Exception as e:
                methods_tried.append(f"USER info (failed: {str(e)[:50]}...)")

            # Add debugging info
            version_info['methods_tried'] = methods_tried
            version_info['connection_status'] = 'Connected successfully'

            return version_info

        finally:
            cursor.close()

# Convenience function for quick access
def get_oracle_connection(config: Optional[OracleConfig] = None):
    """
    Convenience function to get Oracle connection singleton.

    Args:
        config: Optional Oracle configuration

    Returns:
        OracleConnectionSingleton instance
    """
    return OracleConnectionSingleton.get_instance(config)


if __name__ == "__main__":
    # Example usage
    try:
        # Test connection using environment variables
        oracle_conn = get_oracle_connection()

        # Test query
        result = oracle_conn.execute_query("SELECT 'Hello Oracle' FROM dual")
        print(f"Connection test result: {result}")

        # Get database version information
        print("\n=== Oracle Database Version Information ===")
        version_info = oracle_conn.get_database_version()
        for key, value in version_info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

    except Exception as e:
        print(f"Connection test failed: {e}")
        print("\nMake sure to set environment variables:")
        print("- ORACLE_HOST")
        print("- ORACLE_PORT (optional, default: 1521)")
        print("- ORACLE_SERVICE_NAME")
        print("- ORACLE_USER")
        print("- ORACLE_PASSWORD")
        print("\nOr use ORACLE_DSN for complete connection string")