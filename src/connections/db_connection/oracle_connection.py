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
import threading
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from src.connections.db_connection.abstract_db_connection import (
    AbstractDatabaseConnection,
    DatabaseConfig
)
from src.connections.db_connection.query_result import ColumnMetadata
from src.connections.db_connection.security import SecurityMode, SQLValidator

load_dotenv()


import os

try:
    import oracledb

    try:
        if os.name == "nt":
            # ==========================
            # WINDOWS
            # ==========================
            oracle_client_paths = [
                r"C:\Oracle\instantclient_23_9",
                r"C:\Oracle\instantclient_21_3",
                r"C:\Oracle\instantclient",
            ]

            initialized = False
            for path in oracle_client_paths:
                if os.path.isdir(path):
                    try:
                        oracledb.init_oracle_client(lib_dir=path)
                        print(f"[OK] Oracle thick mode initialized (Windows): {path}")
                        initialized = True
                        break
                    except Exception:
                        continue

            if not initialized:
                raise RuntimeError("Oracle Instant Client not found on Windows")

        else:
            # ==========================
            # LINUX / DOCKER
            # ==========================
            # Instant Client già installato
            # LD_LIBRARY_PATH e ldconfig già configurati dal Dockerfile
            oracledb.init_oracle_client()
            print("[OK] Oracle thick mode initialized (Linux/Docker)")

    except Exception as e:
        print(f"[WARN] Thick mode initialization failed: {e}")
        print("[WARN] Falling back to thin mode (requires Oracle DB >= 12.1)")

except ImportError:
    raise ImportError(
        "oracledb package not found. Install with: pip install oracledb"
    )


class ConnectionMethod(Enum):
    """Supported Oracle connection methods"""
    EASY_CONNECT = "easy_connect"  # host:port/service_name
    DSN = "dsn"                   # Complete connection string
    TNS_ALIAS = "tns_alias"       # TNS alias from tnsnames.ora


@dataclass
class OracleConfig(DatabaseConfig):
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
        # Validazione per Easy Connect method
        if self.method == ConnectionMethod.EASY_CONNECT and (not self.host or not self.service_name):
            raise ValueError("host and service_name are required for Easy Connect method")

        # Validazione per DSN method
        if self.method == ConnectionMethod.DSN and not self.dsn:
            raise ValueError("dsn is required for DSN method")

    @classmethod
    def from_k8s_config(cls) -> 'OracleConfig':
        """
        Crea configurazione da K8sConfig (infrastructure.yaml).

        Tutte le configurazioni (host, port, service_name, dsn, user, password)
        vengono lette da infrastructure.yaml.

        Returns:
            OracleConfig configurato

        Raises:
            ValueError: Se mancano credenziali o configurazione database
        """
        from src.k8s_config import YamlConfig
        import os

        # Leggi tutte le configurazioni da infrastructure.yaml (include credenziali)
        infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
        oracle_db_config = infra_config.get("databases.oracle")

        # Se DSN è fornito, usa quello
        if oracle_db_config.get("dsn"):
            return cls(
                user=oracle_db_config["user"],
                password=oracle_db_config["password"],
                dsn=oracle_db_config["dsn"],
                method=ConnectionMethod.DSN
            )

        # Altrimenti usa Easy Connect method
        return cls(
            user=oracle_db_config["user"],
            password=oracle_db_config["password"],
            host=oracle_db_config["host"],
            port=oracle_db_config["port"],
            service_name=oracle_db_config["service_name"],
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


class OracleConnection(AbstractDatabaseConnection[OracleConfig, oracledb.Connection]):
    """
    Thread-safe singleton for Oracle database connections.

    Example usage:
        # Using environment variables
        conn = OracleConnection.get_instance()

        # Using custom configuration
        config = OracleConfig(
            user="myuser",
            password="mypass",
            host="localhost",
            service_name="ORCLPDB"
        )
        conn = OracleConnection.get_instance(config)

        # Execute query (returns QueryResult)
        result = conn.execute_query("SELECT * FROM dual")
        print(result.as_dicts())
    """

    _instance: Optional['OracleConnection'] = None
    _lock: threading.Lock = threading.Lock()
    _connection: Optional[oracledb.Connection] = None
    _config: Optional[OracleConfig] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(
        cls,
        config: Optional[OracleConfig] = None,
        security_mode: SecurityMode = SecurityMode.READ_ONLY,
        custom_validator: Optional[SQLValidator] = None
    ) -> 'OracleConnection':
        """
        Get singleton instance with optional configuration.

        Args:
            config: Oracle configuration. If None, will try to load from environment.
            security_mode: Security mode (default: READ_ONLY for AI safety)
            custom_validator: Custom SQL validator (if security_mode is CUSTOM)

        Returns:
            OracleConnection instance
        """
        instance = cls(security_mode=security_mode, custom_validator=custom_validator)

        # Set configuration if provided or if not already set
        if config is not None or cls._config is None:
            if config is None:
                config = OracleConfig.from_k8s_config()
            cls._config = config
            # Reset connection if config changes
            if cls._connection is not None:
                try:
                    cls._connection.close()
                except Exception:
                    pass
                cls._connection = None

        return instance

    @property
    def param_style(self) -> str:
        """Oracle uses :named parameter style"""
        return 'named'

    def _create_connection(self):
        """Create a new Oracle database connection"""
        if self._config is None:
            raise RuntimeError(
                "Oracle connection not configured. "
                "Call OracleConnection.get_instance() to initialize with environment variables, "
                "or pass an OracleConfig object to get_instance(config=...)."
            )

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
        except Exception:
            return False

    def close_connection(self):
        """Close the database connection"""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except Exception:
                    pass
                finally:
                    self._connection = None

    def _execute_raw(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Tuple[Any, ...]], List[ColumnMetadata]]:
        """
        Execute query without validation (internal use).

        Args:
            query: SQL query string
            params: Query parameters (dict with :named style)

        Returns:
            Tuple of (rows, column_metadata)
        """
        connection = self.get_connection()
        cursor = connection.cursor()

        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            results = cursor.fetchall()

            # Convert cursor.description to ColumnMetadata
            columns = []
            if cursor.description:
                for desc in cursor.description:
                    columns.append(ColumnMetadata(
                        name=desc[0],
                        type_code=desc[1],
                        display_size=desc[2],
                        precision=desc[4] if len(desc) > 4 else None,
                        scale=desc[5] if len(desc) > 5 else None,
                        nullable=desc[6] if len(desc) > 6 else None
                    ))

            return results, columns
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
def get_oracle_connection(
    config: Optional[OracleConfig] = None,
    security_mode: SecurityMode = SecurityMode.READ_ONLY
) -> OracleConnection:
    """
    Convenience function to get Oracle connection singleton.

    Args:
        config: Optional Oracle configuration
        security_mode: Security mode (default: READ_ONLY)

    Returns:
        OracleConnection instance
    """
    return OracleConnection.get_instance(config, security_mode=security_mode)
