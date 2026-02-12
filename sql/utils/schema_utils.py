# src/utils/schema_utils.py
"""
Utility per generare rappresentazione SQL dello schema dalla cache JSON.
"""
import json
from pathlib import Path
from typing import Dict, List, Any


def load_schema_cache(cache_path: Path) -> Dict[str, Any]:
    """Carica la cache JSON dello schema."""
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_sql_schema(cache_path: Path) -> str:
    """
    Genera rappresentazione SQL CREATE TABLE/VIEW dallo schema cache.

    Returns:
        Stringa con CREATE statements per ogni tabella/vista.
    """
    schema_data = load_schema_cache(cache_path)

    statements = []
    for table in schema_data.get("tables", []):
        table_name = table["table_name"]
        table_type = table.get("type", "TABLE")
        description = table.get("description", "")

        columns_def = []
        for col in table.get("columns", []):
            col_name = col["column_name"]
            col_type = col.get("type", "VARCHAR2(255)")
            col_desc = col.get("description", "")

            col_line = f"    {col_name} {col_type}"
            if col.get("is_primary_key"):
                col_line += " PRIMARY KEY"
            if col_desc:
                col_line += f"  -- {col_desc}"
            columns_def.append(col_line)

        stmt = f"-- {description}\nCREATE {table_type} {table_name} (\n"
        stmt += ",\n".join(columns_def)
        stmt += "\n);"
        statements.append(stmt)

    return "\n\n".join(statements)


def get_table_columns(cache_path: Path) -> Dict[str, List[str]]:
    """
    Estrae mapping tabella -> lista colonne dalla cache.
    """
    schema_data = load_schema_cache(cache_path)
    return {
        table["table_name"]: [col["column_name"] for col in table.get("columns", [])]
        for table in schema_data.get("tables", [])
    }
