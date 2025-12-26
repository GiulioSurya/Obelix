# Database Connection Module

This module provides a unified interface for database connections in the Obelix framework. It enables interchangeable use of different database backends (Oracle, PostgreSQL) with consistent behavior and configurable security.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Security Modes](#security-modes)
- [Query Results](#query-results)
- [Oracle Connection](#oracle-connection)
- [PostgreSQL Connection](#postgresql-connection)
- [Using with AI Tools](#using-with-ai-tools)
- [API Reference](#api-reference)

## Overview

The database connection module solves the problem of database interchangeability in AI agent frameworks. A tool written for Oracle works identically with PostgreSQL by simply swapping the connection object.

Key features:

- **Uniform Results**: All databases return `QueryResult` objects with the same interface
- **Configurable Security**: Control allowed SQL operations via `SecurityMode`
- **Thread-safe Singleton**: Each connection type maintains a single instance
- **Connection Pooling**: PostgreSQL uses connection pooling for efficiency

## Architecture

```
AbstractDatabaseConnection
    |
    +-- OracleConnection
    |       Uses: oracledb
    |       Param style: :named
    |
    +-- PostgresConnection
            Uses: psycopg + psycopg_pool
            Param style: %s positional
```

### File Structure

```
db_connection/
    __init__.py              # Public exports
    abstract_db_connection.py # Base class
    query_result.py          # QueryResult, ColumnMetadata
    security.py              # SecurityMode, SQLValidator
    oracle_connection.py     # Oracle implementation
    postgres_connection.py   # PostgreSQL implementation
    README.md                # This file
```

## Installation

### Oracle

```bash
pip install oracledb
```

For thick mode (required for some Oracle features), install Oracle Instant Client:
- Windows: Place in `C:\Oracle\instantclient_*`
- Linux: Configure `LD_LIBRARY_PATH`

### PostgreSQL

```bash
pip install psycopg[binary,pool]
```

## Quick Start

### Basic Usage

```python
from src.connections.db_connection import (
    OracleConnection,
    PostgresConnection,
    OracleConfig,
    PostgresConfig,
    SecurityMode
)

# Oracle
oracle_config = OracleConfig(
    user="myuser",
    password="mypass",
    host="localhost",
    service_name="ORCLPDB"
)
oracle_conn = OracleConnection.get_instance(oracle_config)

# PostgreSQL
pg_config = PostgresConfig(
    host="localhost",
    port=5432,
    database="mydb",
    user="myuser",
    password="mypass"
)
pg_conn = PostgresConnection.get_instance(pg_config)

# Both work the same way
result = oracle_conn.execute_query("SELECT * FROM users WHERE id = :id", {"id": 1})
result = pg_conn.execute_query("SELECT * FROM users WHERE id = %s", (1,))

# Access results uniformly
for row in result.as_dicts():
    print(row["name"])
```

### Using Configuration Files

Both connections support loading configuration from `infrastructure.yaml`:

```python
# Loads from INFRASTRUCTURE_CONFIG_PATH environment variable
oracle_conn = OracleConnection.get_instance()
pg_conn = PostgresConnection.get_instance()
```

## Security Modes

Security modes control which SQL operations are allowed. This is critical for AI agents that generate SQL.

| Mode | Allowed Operations | Use Case |
|------|-------------------|----------|
| `READ_ONLY` | SELECT, WITH | AI agents (default) |
| `NONE` | All operations | Trusted internal code |
| `CUSTOM` | User-defined | Custom validation logic |

### Example: Read-Only Mode (Default)

```python
conn = OracleConnection.get_instance(
    config,
    security_mode=SecurityMode.READ_ONLY
)

# This works
result = conn.execute_query("SELECT * FROM users")

# This raises SQLSecurityException
result = conn.execute_query("DELETE FROM users")
```

### Example: Full Access Mode

```python
conn = PostgresConnection.get_instance(
    config,
    security_mode=SecurityMode.NONE
)

# All operations allowed
conn.execute_query("INSERT INTO logs VALUES (...)")
```

### Example: Custom Validator

```python
from src.connections.db_connection import SQLValidator, SecurityMode

class MyValidator(SQLValidator):
    def validate(self, query: str) -> None:
        # Custom validation logic
        if "DROP" in query.upper():
            raise SQLSecurityException("DROP not allowed")

conn = OracleConnection.get_instance(
    config,
    security_mode=SecurityMode.CUSTOM,
    custom_validator=MyValidator()
)
```

## Query Results

All `execute_query()` calls return a `QueryResult` object with a consistent interface.

### QueryResult Properties

| Property | Type | Description |
|----------|------|-------------|
| `status` | `QueryStatus` | SUCCESS, ERROR, or EMPTY |
| `rows` | `List[Tuple]` | Raw result rows |
| `columns` | `List[ColumnMetadata]` | Column metadata |
| `row_count` | `int` | Number of rows |
| `is_success` | `bool` | True if query succeeded |
| `error` | `Optional[str]` | Error message if failed |

### QueryResult Methods

```python
result = conn.execute_query("SELECT id, name FROM users")

# Get column names
print(result.column_names)  # ['id', 'name']

# Convert to list of dictionaries
users = result.as_dicts()
# [{'id': 1, 'name': 'John'}, {'id': 2, 'name': 'Jane'}]

# Get first row as dict
user = result.first()
# {'id': 1, 'name': 'John'}

# Get single scalar value
count = conn.execute_query("SELECT COUNT(*) FROM users").scalar()
# 42
```

## Oracle Connection

### Configuration

```python
from src.connections.db_connection import OracleConfig, ConnectionMethod

# Easy Connect (default)
config = OracleConfig(
    user="myuser",
    password="mypass",
    host="localhost",
    port=1521,
    service_name="ORCLPDB",
    method=ConnectionMethod.EASY_CONNECT
)

# DSN string
config = OracleConfig(
    user="myuser",
    password="mypass",
    dsn="(DESCRIPTION=(ADDRESS=...))",
    method=ConnectionMethod.DSN
)

# TNS alias
config = OracleConfig(
    user="myuser",
    password="mypass",
    dsn="MYDB_ALIAS",
    method=ConnectionMethod.TNS_ALIAS
)
```

### Parameter Style

Oracle uses named parameters with `:param` syntax:

```python
result = conn.execute_query(
    "SELECT * FROM users WHERE id = :user_id AND status = :status",
    {"user_id": 123, "status": "active"}
)
```

## PostgreSQL Connection

### Configuration

```python
from src.connections.db_connection import PostgresConfig

config = PostgresConfig(
    host="localhost",
    port=5432,
    database="mydb",
    user="myuser",
    password="mypass",
    embed_dim=1024  # For vector operations
)
```

### Parameter Style

PostgreSQL uses positional parameters with `%s` syntax:

```python
result = conn.execute_query(
    "SELECT * FROM users WHERE id = %s AND status = %s",
    (123, "active")
)
```

### Additional Operations

PostgreSQL connection includes DDL/DML helper methods:

```python
# Create table
conn.create_table("users", {
    "id": "SERIAL PRIMARY KEY",
    "name": "VARCHAR(100)",
    "email": "VARCHAR(255) UNIQUE"
})

# Insert data
user_id = conn.insert_data(
    "users",
    {"name": "John", "email": "john@example.com"},
    returning="id"
)

# Check existence
if conn.table_exists("users"):
    print("Table exists")

# Enable extensions (e.g., pgvector)
conn.enable_extension("vector")
```

Note: DDL/DML methods bypass security validation and are intended for administrative use, not AI-generated queries.

## Using with AI Tools

The primary use case is AI agents that generate and execute SQL queries.

### SqlQueryExecutorTool

```python
from src.tools.sql_query_executor_tool import SqlQueryExecutorTool
from src.connections.db_connection import get_oracle_connection, get_postgres_connection

# Create tool with Oracle
oracle_tool = SqlQueryExecutorTool(get_oracle_connection())

# Create tool with PostgreSQL
pg_tool = SqlQueryExecutorTool(get_postgres_connection())

# Both tools work identically
# The LLM is instructed in the system prompt which syntax to use
```

### System Prompt Example

```
You are a SQL assistant connected to an Oracle database.
Use :param_name syntax for query parameters.
Example: SELECT * FROM users WHERE id = :user_id
```

Or for PostgreSQL:

```
You are a SQL assistant connected to a PostgreSQL database.
Use %s syntax for query parameters.
Example: SELECT * FROM users WHERE id = %s
```

## API Reference

### AbstractDatabaseConnection

| Method | Description |
|--------|-------------|
| `get_instance(config, security_mode)` | Get singleton instance |
| `execute_query(query, params)` | Execute query, return QueryResult |
| `execute_query_dict(query, params)` | Execute query, return List[Dict] |
| `get_connection()` | Get raw database connection |
| `close_connection()` | Close connection/pool |
| `get_database_version()` | Get database version info |
| `param_style` | Property: 'named' or 'positional' |

### SecurityMode

| Value | Description |
|-------|-------------|
| `READ_ONLY` | Only SELECT/WITH allowed (default) |
| `NONE` | All operations allowed |
| `CUSTOM` | Use custom validator |

### QueryStatus

| Value | Description |
|-------|-------------|
| `SUCCESS` | Query executed with results |
| `EMPTY` | Query executed, no results |
| `ERROR` | Query failed |
