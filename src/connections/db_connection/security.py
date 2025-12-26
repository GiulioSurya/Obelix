"""
SQL Security module for database connections.

Provides configurable SQL validation to control which operations
are allowed on database connections.
"""

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional


class SecurityMode(str, Enum):
    """Security modes for database connections"""
    NONE = "none"           # No validation (user takes responsibility)
    READ_ONLY = "read_only" # Only SELECT/WITH allowed (default for AI tools)
    CUSTOM = "custom"       # Custom validator provided


class SQLSecurityException(Exception):
    """Exception raised when a dangerous SQL query is detected"""
    pass


class SQLValidator(ABC):
    """Abstract base class for SQL validators"""

    @abstractmethod
    def validate(self, query: str) -> None:
        """
        Validate SQL query.

        Args:
            query: SQL query string to validate

        Raises:
            SQLSecurityException: If query is not allowed
            ValueError: If query is empty
        """
        pass


class NoOpValidator(SQLValidator):
    """Validator that does nothing (SecurityMode.NONE)"""

    def validate(self, query: str) -> None:
        if not query or not query.strip():
            raise ValueError("SQL query cannot be empty")


class ReadOnlySQLValidator(SQLValidator):
    """
    SQL validator that allows only SELECT and WITH (CTE) statements.

    Blocks all potentially dangerous operations:
    - DML: INSERT, UPDATE, DELETE, MERGE, TRUNCATE
    - DDL: CREATE, ALTER, DROP, RENAME
    - Privileges: GRANT, REVOKE
    - Execution: EXECUTE, EXEC, CALL
    - Transaction: COMMIT, ROLLBACK, SAVEPOINT
    - Locking: LOCK

    Also detects multi-statement attacks.
    """

    DANGEROUS_KEYWORDS = frozenset([
        'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'MERGE',  # DML
        'DROP', 'CREATE', 'ALTER', 'RENAME',  # DDL
        'GRANT', 'REVOKE',  # Privileges
        'EXECUTE', 'EXEC', 'CALL',  # Execution
        'COMMIT', 'ROLLBACK', 'SAVEPOINT',  # Transaction
        'LOCK',  # Locking
    ])

    def validate(self, query: str) -> None:
        """
        Validate SQL query for read-only operations.

        Args:
            query: SQL query string to validate

        Raises:
            ValueError: If query is empty
            SQLSecurityException: If query contains dangerous operations
        """
        if not query or not query.strip():
            raise ValueError("SQL query cannot be empty")

        # Remove comments for validation
        query_no_comments = self._remove_comments(query)

        # Check for multi-statement attacks
        if self._detect_multi_statements(query_no_comments):
            raise SQLSecurityException(
                "Multiple SQL statements detected. Only single SELECT or WITH queries are allowed."
            )

        # Remove strings to avoid false positives
        query_no_strings = self._remove_strings(query_no_comments)

        # Normalize whitespace and convert to uppercase
        query_normalized = ' '.join(query_no_strings.split()).upper()

        # Check for dangerous keywords
        for keyword in self.DANGEROUS_KEYWORDS:
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, query_normalized):
                raise SQLSecurityException(
                    f"Dangerous SQL keyword detected: {keyword}. "
                    "Only SELECT and WITH queries are allowed."
                )

        # Check if query starts with SELECT or WITH
        query_stripped = query_normalized.strip()
        while query_stripped.startswith('('):
            query_stripped = query_stripped[1:].strip()

        if not (query_stripped.startswith('SELECT') or query_stripped.startswith('WITH')):
            raise SQLSecurityException(
                "Query must start with SELECT or WITH. Only read-only queries are allowed."
            )

    @staticmethod
    def _remove_comments(query: str) -> str:
        """Remove SQL comments from query"""
        # Remove -- line comments
        query = re.sub(r'--[^\n]*', ' ', query)
        # Remove /* block comments */
        query = re.sub(r'/\*.*?\*/', ' ', query, flags=re.DOTALL)
        return query

    @staticmethod
    def _remove_strings(query: str) -> str:
        """Replace string literals with placeholders"""
        query = re.sub(r"'[^']*'", "'STRING'", query)
        query = re.sub(r'"[^"]*"', '"STRING"', query)
        return query

    @staticmethod
    def _detect_multi_statements(query: str) -> bool:
        """Detect multiple statements separated by semicolons"""
        query_no_strings = ReadOnlySQLValidator._remove_strings(query)
        parts = [p.strip() for p in query_no_strings.split(';') if p.strip()]
        return len(parts) > 1


def create_validator(
    mode: SecurityMode,
    custom_validator: Optional[SQLValidator] = None
) -> SQLValidator:
    """
    Factory function to create appropriate validator.

    Args:
        mode: Security mode
        custom_validator: Custom validator (required if mode is CUSTOM)

    Returns:
        SQLValidator instance
    """
    if mode == SecurityMode.CUSTOM:
        if custom_validator is None:
            raise ValueError("custom_validator required when mode is CUSTOM")
        return custom_validator
    elif mode == SecurityMode.READ_ONLY:
        return ReadOnlySQLValidator()
    else:
        return NoOpValidator()
