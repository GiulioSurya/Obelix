"""
Oracle Database Connection Pool (Async)
=======================================

Factory functions e helpers per gestire pool di connessioni Oracle async.
Il pool viene creato nel lifespan di FastAPI e passato via dependency injection.

Architettura:
- create_oracle_pool(): Factory per creare pool async (chiamare nel lifespan)
- close_oracle_pool(): Chiude il pool (chiamare nello shutdown)
- execute_query(): Helper async per eseguire query SELECT
- get_sync_connection(): Connessione singola sync per script offline

Requirements:
    pip install oracledb

Environment Variables (via infrastructure.yaml):
    databases.oracle.host
    databases.oracle.port
    databases.oracle.service_name
    databases.oracle.user
    databases.oracle.password
    databases.oracle.pool.min
    databases.oracle.pool.max
    databases.oracle.pool.increment
"""

import os
import re
from dataclasses import dataclass
import asyncio
from contextlib import contextmanager
from typing import Optional, Dict, Any

from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# Oracle Client Initialization (thick/thin mode)
# =============================================================================

try:
    import oracledb

    try:
        if os.name == "nt":
            # Windows
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
                        logger.info(f"Oracle thick mode initialized (Windows): {path}")
                        initialized = True
                        break
                    except Exception:
                        continue

            if not initialized:
                raise RuntimeError("Oracle Instant Client not found on Windows")

        else:
            # Linux / Docker
            oracledb.init_oracle_client()
            logger.info("Oracle thick mode initialized (Linux/Docker)")

    except Exception as e:
        logger.warning(f"Thick mode initialization failed: {e}")
        logger.warning("Falling back to thin mode (requires Oracle DB >= 12.1)")

except ImportError:
    raise ImportError(
        "oracledb package not found. Install with: pip install oracledb"
    )


# =============================================================================
# Security: SQL Validator
# =============================================================================

class SQLSecurityException(Exception):
    """Exception raised when a dangerous SQL query is detected."""
    pass


class SQLValidator:
    """
    SQL query validator that allows only SELECT and WITH (CTE) statements.
    Blocks all potentially dangerous operations.
    """

    DANGEROUS_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'MERGE',
        'DROP', 'CREATE', 'ALTER', 'RENAME',
        'GRANT', 'REVOKE',
        'EXECUTE', 'EXEC', 'CALL',
        'COMMIT', 'ROLLBACK', 'SAVEPOINT',
        'LOCK',
    ]

    @staticmethod
    def _remove_comments(query: str) -> str:
        """Remove SQL comments from query."""
        query = re.sub(r'--[^\n]*', ' ', query)
        query = re.sub(r'/\*.*?\*/', ' ', query, flags=re.DOTALL)
        return query

    @staticmethod
    def _remove_strings(query: str) -> str:
        """Replace string literals with placeholders."""
        query = re.sub(r"'[^']*'", "'STRING'", query)
        query = re.sub(r'"[^"]*"', '"STRING"', query)
        return query

    @staticmethod
    def _detect_multi_statements(query: str) -> bool:
        """Detect multiple statements separated by semicolons."""
        query_without_strings = SQLValidator._remove_strings(query)
        parts = [part.strip() for part in query_without_strings.split(';') if part.strip()]
        return len(parts) > 1

    @classmethod
    def validate(cls, query: str) -> None:
        """
        Validate SQL query for security.

        Raises:
            ValueError: If query is empty
            SQLSecurityException: If query contains dangerous operations
        """
        if not query or not query.strip():
            raise ValueError("SQL query cannot be empty")

        query_no_comments = cls._remove_comments(query)

        if cls._detect_multi_statements(query_no_comments):
            raise SQLSecurityException(
                "Multiple SQL statements detected. Only single SELECT or WITH queries are allowed."
            )

        query_no_strings = cls._remove_strings(query_no_comments)
        query_normalized = ' '.join(query_no_strings.split()).upper()

        for keyword in cls.DANGEROUS_KEYWORDS:
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, query_normalized):
                raise SQLSecurityException(
                    f"Dangerous SQL keyword detected: {keyword}. Only SELECT and WITH queries are allowed."
                )

        query_stripped = query_normalized.strip()
        while query_stripped.startswith('('):
            query_stripped = query_stripped[1:].strip()

        if not (query_stripped.startswith('SELECT') or query_stripped.startswith('WITH')):
            raise SQLSecurityException(
                "Query must start with SELECT or WITH. Only read-only queries are allowed."
            )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OraclePoolConfig:
    """Configurazione per Oracle connection pool."""
    user: str
    password: str
    dsn: str
    pool_min: int = 2
    pool_max: int = 10
    pool_increment: int = 1

    @classmethod
    def from_k8s_config(cls) -> 'OraclePoolConfig':
        """Carica configurazione da infrastructure.yaml."""
        from obelix.infrastructure.k8s import YamlConfig

        config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
        oracle_cfg = config.get("databases.oracle")
        pool_cfg = oracle_cfg.get("pool", {})

        dsn = f"{oracle_cfg['host']}:{oracle_cfg['port']}/{oracle_cfg['service_name']}"

        return cls(
            user=oracle_cfg["user"],
            password=oracle_cfg["password"],
            dsn=dsn,
            pool_min=pool_cfg.get("min", 2),
            pool_max=pool_cfg.get("max", 10),
            pool_increment=pool_cfg.get("increment", 1)
        )


# =============================================================================
# Pool Factory Functions (sync pool - per API)
# =============================================================================

def create_oracle_pool(
    config: Optional[OraclePoolConfig] = None
) -> oracledb.ConnectionPool:
    """
    Crea un pool di connessioni Oracle (sync).

    Chiamare nel lifespan di FastAPI, salvare in app.state.

    Args:
        config: Configurazione pool. Se None, carica da infrastructure.yaml.

    Returns:
        Pool di connessioni pronto all'uso.
    """
    if config is None:
        config = OraclePoolConfig.from_k8s_config()

    pool = oracledb.create_pool(
        user=config.user,
        password=config.password,
        dsn=config.dsn,
        min=config.pool_min,
        max=config.pool_max,
        increment=config.pool_increment
    )

    logger.info(f"Oracle pool created: min={config.pool_min}, max={config.pool_max}")
    return pool


def close_oracle_pool(pool: oracledb.ConnectionPool) -> None:
    """
    Chiude il pool di connessioni.

    Chiamare nello shutdown del lifespan.
    """
    if pool is not None:
        pool.close()
        logger.info("Oracle pool closed")


# =============================================================================
# Connection Helpers (async wrapper over sync pool)
# =============================================================================

@contextmanager
def acquire_connection(pool: oracledb.ConnectionPool):
    """
    Context manager per acquisire/rilasciare connessione dal pool.

    Usage:
        with acquire_connection(pool) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
    """
    connection = pool.acquire()
    try:
        yield connection
    finally:
        # In oracledb, closing a pooled connection returns it to the pool.
        try:
            connection.close()
        except Exception:
            pass


async def execute_query(
    pool: oracledb.ConnectionPool,
    query: str,
    params: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Esegue una query SELECT e ritorna (results, cursor.description).

    SECURITY: Valida la query prima dell'esecuzione (solo SELECT/WITH ammessi).

    Args:
        pool: Pool di connessioni Oracle (sync)
        query: Query SQL
        params: Parametri query (opzionali)

    Returns:
        Tuple (results, cursor.description)

    Raises:
        SQLSecurityException: Se la query contiene operazioni pericolose
        ValueError: Se la query e vuota
    """
    def _execute_query_sync() -> tuple:
        SQLValidator.validate(query)
        with acquire_connection(pool) as connection:
            cursor = connection.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                results = cursor.fetchall()
                description = cursor.description
                return (results, description)
            finally:
                cursor.close()

    # oracledb in thick mode is sync: run in threadpool to avoid blocking FastAPI.
    return await asyncio.to_thread(_execute_query_sync)


# =============================================================================
# Sync Connection (per script offline come DBSchemaExtractor)
# =============================================================================

def get_sync_connection(
    config: Optional[OraclePoolConfig] = None
) -> oracledb.Connection:
    """
    Crea una connessione Oracle sincrona singola.

    Usare SOLO per script offline (es. DBSchemaExtractor, cache generation).
    NON usare nel flusso API - usare il pool async invece.

    Args:
        config: Configurazione. Se None, carica da infrastructure.yaml.

    Returns:
        Connessione Oracle sincrona.
    """
    if config is None:
        config = OraclePoolConfig.from_k8s_config()

    return oracledb.connect(
        user=config.user,
        password=config.password,
        dsn=config.dsn
    )


def execute_query_sync(
    connection: oracledb.Connection,
    query: str,
    params: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Esegue una query SELECT in modo sincrono.

    Usare SOLO per script offline.

    Args:
        connection: Connessione Oracle sincrona
        query: Query SQL
        params: Parametri query (opzionali)

    Returns:
        Tuple (results, cursor.description)
    """
    SQLValidator.validate(query)

    cursor = connection.cursor()
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        results = cursor.fetchall()
        description = cursor.description
        return (results, description)
    finally:
        cursor.close()


# =============================================================================
# Test / Main
# =============================================================================

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    async def pool():
        """Test del pool async."""
        logger.info("=== Testing Oracle Async Pool ===")

        pool = create_oracle_pool()

        try:
            # Test query
            results, desc = await execute_query(pool, """WITH base AS (
   SELECT CAPITOLO, DESCRIZIONE_CAP,
          MISSIONE, DES_MISSIONE,
          PROGRAMMA_ARM, DES_PROGRAMMA_ARM,
          ESERCIZIO,
          ROUND(NVL(STN_INIZIALE_CO,0),2) AS stanziamento
   FROM VISTA_BILANCIO_SPESA_AI
   WHERE TIPO = 'E'
     AND MISSIONE = '03'
     AND UPPER(DES_PROGRAMMA_ARM) LIKE UPPER('%POLIZIA LOCALE E AMMINISTRATIVA%')
     AND ESERCIZIO BETWEEN 2020 AND 2025
),
aggregati AS (
   SELECT CAPITOLO, DESCRIZIONE_CAP,
          MISSIONE, DES_MISSIONE,
          PROGRAMMA_ARM, DES_PROGRAMMA_ARM,
          MAX(CASE WHEN ESERCIZIO = 2025 THEN stanziamento END) AS stan_2025,
          AVG(CASE WHEN ESERCIZIO BETWEEN 2020 AND 2024 THEN stanziamento END) AS stan_media_5_anni,
          MAX(CASE WHEN ESERCIZIO = 2025 THEN stanziamento END) -
          AVG(CASE WHEN ESERCIZIO BETWEEN 2020 AND 2024 THEN stanziamento END) AS diff_2025_vs_media,
          MAX(CASE WHEN ESERCIZIO = 2024 THEN stanziamento END) AS stan_2024,
          MAX(CASE WHEN ESERCIZIO = 2023 THEN stanziamento END) AS stan_2023,
          MAX(CASE WHEN ESERCIZIO = 2022 THEN stanziamento END) AS stan_2022,
          MAX(CASE WHEN ESERCIZIO = 2021 THEN stanziamento END) AS stan_2021,
          MAX(CASE WHEN ESERCIZIO = 2020 THEN stanziamento END) AS stan_2020
   FROM base
   GROUP BY CAPITOLO, DESCRIZIONE_CAP,
            MISSIONE, DES_MISSIONE,
            PROGRAMMA_ARM, DES_PROGRAMMA_ARM
)
SELECT CAPITOLO, DESCRIZIONE_CAP,
       MISSIONE, DES_MISSIONE,
       PROGRAMMA_ARM, DES_PROGRAMMA_ARM,
       stan_2025, stan_media_5_anni, diff_2025_vs_media,
       stan_2024, stan_2023, stan_2022, stan_2021, stan_2020
FROM aggregati
ORDER BY CAPITOLO""")
            logger.info(f"Test query result: {results}")

            # Test concurrent queries
            async def run_query(n):
                results, _ = await execute_query(pool, f"SELECT {n} FROM DUAL")
                return results[0][0]

            tasks = [run_query(i) for i in range(5)]
            concurrent_results = await asyncio.gather(*tasks)
            logger.info(f"Concurrent results: {concurrent_results}")

        finally:
            close_oracle_pool(pool)

    asyncio.run(pool())
