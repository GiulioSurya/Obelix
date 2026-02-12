# src/database/schema/literal_generator.py
"""
Generatore di Literal Python dalla cache JSON dello schema.
"""
import json
from pathlib import Path
from datetime import datetime


def generate_literals_from_cache(cache_path: Path, output_path: Path) -> None:
    """
    Genera file Python con Literal a partire dalla cache JSON.
    """
    with open(cache_path, 'r', encoding='utf-8') as f:
        schema_data = json.load(f)

    tables = schema_data.get("tables", [])
    if not tables:
        raise ValueError(f"Nessuna tabella trovata in {cache_path}")

    # Estrai struttura
    schema_structure = {
        table["table_name"]: [col["column_name"] for col in table.get("columns", [])]
        for table in tables
    }

    # Genera Literal per colonne
    column_literals = []
    for table_name, columns in schema_structure.items():
        type_name = _to_pascal_case(table_name) + 'Columns'
        cols_str = ', '.join(f'"{c}"' for c in columns)
        column_literals.append(f'{type_name} = Literal[{cols_str}]')

    # Genera ALL_COLUMNS dict
    all_columns_items = []
    for table_name, columns in schema_structure.items():
        cols_list = ', '.join(f'"{c}"' for c in columns)
        all_columns_items.append(f'    "{table_name}": [{cols_list}],')

    table_names = list(schema_structure.keys())
    table_literal = ', '.join(f'"{t}"' for t in table_names)

    content = f'''# AUTO-GENERATED - Non modificare manualmente
# Generato: {datetime.now().isoformat()}
# Sorgente: {cache_path.name}

from typing import Literal, Dict, List

TableName = Literal[{table_literal}]

{chr(10).join(column_literals)}

ALL_TABLES: List[str] = {table_names}

ALL_COLUMNS: Dict[str, List[str]] = {{
{chr(10).join(all_columns_items)}
}}
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def _to_pascal_case(snake_str: str) -> str:
    components = snake_str.lower().split('_')
    return ''.join(x.title() for x in components)
