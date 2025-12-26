"""
Query Result wrapper for uniform database results.

Provides a consistent interface for query results across different
database backends (Oracle, PostgreSQL, etc.).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class QueryStatus(str, Enum):
    """Status of query execution"""
    SUCCESS = "success"
    ERROR = "error"
    EMPTY = "empty"  # Query executed but no results


@dataclass
class ColumnMetadata:
    """Metadata for a result column"""
    name: str
    type_code: Optional[Any] = None
    display_size: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    nullable: Optional[bool] = None


@dataclass
class QueryResult:
    """
    Uniform result wrapper for database queries.

    Works with both Oracle and PostgreSQL, providing consistent
    access to query results regardless of the underlying database.

    Example:
        result = conn.execute_query("SELECT id, name FROM users")

        # Access as tuples
        for row in result.rows:
            print(row)

        # Access as dicts
        for row in result.as_dicts():
            print(row['name'])

        # Get first result
        user = result.first()

        # Get single value
        count = conn.execute_query("SELECT COUNT(*) FROM users").scalar()
    """
    status: QueryStatus
    rows: List[Tuple[Any, ...]] = field(default_factory=list)
    columns: List[ColumnMetadata] = field(default_factory=list)
    rows_affected: int = 0
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None

    @property
    def column_names(self) -> List[str]:
        """List of column names"""
        return [col.name for col in self.columns]

    @property
    def row_count(self) -> int:
        """Number of rows in result"""
        return len(self.rows)

    @property
    def is_empty(self) -> bool:
        """True if no results"""
        return len(self.rows) == 0

    @property
    def is_success(self) -> bool:
        """True if query executed successfully"""
        return self.status in (QueryStatus.SUCCESS, QueryStatus.EMPTY)

    def as_dicts(self) -> List[Dict[str, Any]]:
        """Convert rows to list of dictionaries"""
        if not self.columns:
            return []
        names = self.column_names
        return [dict(zip(names, row)) for row in self.rows]

    def first(self) -> Optional[Dict[str, Any]]:
        """Return first row as dict, or None if empty"""
        dicts = self.as_dicts()
        return dicts[0] if dicts else None

    def scalar(self) -> Optional[Any]:
        """Return first value of first row, or None if empty"""
        if self.rows and self.rows[0]:
            return self.rows[0][0]
        return None

    @classmethod
    def success(
        cls,
        rows: List[Tuple[Any, ...]],
        columns: List[ColumnMetadata],
        execution_time_ms: Optional[float] = None
    ) -> 'QueryResult':
        """Factory for successful query result"""
        status = QueryStatus.SUCCESS if rows else QueryStatus.EMPTY
        return cls(
            status=status,
            rows=rows,
            columns=columns,
            rows_affected=len(rows),
            execution_time_ms=execution_time_ms
        )

    @classmethod
    def error(
        cls,
        error_msg: str,
        execution_time_ms: Optional[float] = None
    ) -> 'QueryResult':
        """Factory for error result"""
        return cls(
            status=QueryStatus.ERROR,
            error=error_msg,
            execution_time_ms=execution_time_ms
        )
