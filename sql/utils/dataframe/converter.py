"""
DataFrame Converter Utility
============================

Utility per convertire l'output strutturato del SqlQueryExecutorTool
in pandas DataFrame.

Gestisce:
- Conversione diretta da output tool a DataFrame
- Validazione struttura dati
- Gestione valori NULL/None
- Preservazione tipi di dati Oracle (date, decimal, etc.)
"""

from typing import Dict, Any, Optional, List
import pandas as pd
from dotenv import load_dotenv

from src.core.model import ToolResult, ToolStatus
from sql.utils.dataframe.year_detector import YearDetector
from src.infrastructure.k8s import YamlConfig
import os
load_dotenv()

infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
FORCED_CATEGORICAL_COLUMNS: List[str] = infra_config.get("dataframe.forced_categorical_columns")


class DataFrameConverter:
    """
    Converte output di SqlQueryExecutorTool in pandas DataFrame.

    Esempio:
        >>> tool_result = ToolResult(
        ...     result={
        ...         "columns": ["ID", "NAME", "EMAIL"],
        ...         "data": [(1, "Alice", "alice@example.com"), (2, "Bob", "bob@example.com")],
        ...         "row_count": 2
        ...     },
        ...     status=ToolStatus.SUCCESS
        ... )
        >>> df = DataFrameConverter.from_tool_result(tool_result)
        >>> print(df.shape)
        (2, 3)
    """

    # Detector per rilevare e convertire colonne anno
    _year_detector = YearDetector()

    @staticmethod
    def from_tool_result(tool_result: ToolResult) -> pd.DataFrame:
        """
        Converte ToolResult in DataFrame pandas.

        Args:
            tool_result: Risultato del tool con struttura {columns, data, row_count}

        Returns:
            pandas DataFrame con i dati

        Raises:
            ValueError: Se il tool result ha status ERROR o struttura invalida
            TypeError: Se result non è un dict o mancano campi richiesti
        """
        # Verifica status del tool
        if tool_result.status == ToolStatus.ERROR:
            raise ValueError(
                f"Cannot convert ERROR tool result to DataFrame. "
                f"Error: {tool_result.error}"
            )

        # Verifica che result non sia None
        if tool_result.result is None:
            raise ValueError("Tool result is None, cannot convert to DataFrame")

        # Verifica che result sia un dict
        if not isinstance(tool_result.result, dict):
            raise TypeError(
                f"Expected result to be dict, got {type(tool_result.result).__name__}"
            )

        return DataFrameConverter.from_dict(tool_result.result)

    @staticmethod
    def from_dict(result_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Converte dizionario strutturato in DataFrame pandas.

        Args:
            result_dict: Dict con chiavi 'columns', 'data', 'row_count'

        Returns:
            pandas DataFrame

        Raises:
            KeyError: Se mancano campi richiesti
            ValueError: Se i dati sono inconsistenti
        """
        # Verifica campi richiesti
        required_fields = {"columns", "data", "row_count"}
        missing_fields = required_fields - set(result_dict.keys())

        if missing_fields:
            raise KeyError(
                f"Missing required fields in result dict: {missing_fields}"
            )

        columns = result_dict["columns"]
        data = result_dict["data"]
        row_count = result_dict["row_count"]

        # Validazione: row_count deve corrispondere a len(data)
        if len(data) != row_count:
            raise ValueError(
                f"Inconsistent data: row_count={row_count} but data has {len(data)} rows"
            )

        # Caso speciale: risultato vuoto
        if row_count == 0:
            # Crea DataFrame vuoto ma con le colonne corrette
            return pd.DataFrame(columns=columns)

        # Validazione: ogni riga deve avere lo stesso numero di elementi delle colonne
        for idx, row in enumerate(data):
            if len(row) != len(columns):
                raise ValueError(
                    f"Row {idx} has {len(row)} values but {len(columns)} columns expected"
                )

        # Converti in DataFrame
        # Nota: pandas gestisce automaticamente tipi Oracle comuni (date, Decimal, etc.)
        df = pd.DataFrame(data, columns=columns)

        # Converti colonne anno in datetime
        df = DataFrameConverter._convert_year_columns(df)

        # Converti colonne forzate a categoriche
        df = DataFrameConverter._convert_categorical_columns(df)

        return df

    @staticmethod
    def _convert_year_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rileva e converte colonne anno in datetime.

        Args:
            df: DataFrame da processare

        Returns:
            DataFrame con colonne anno convertite in datetime
        """
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                result = DataFrameConverter._year_detector.detect(df[col], col)
                if result.is_year:
                    # Converti anno intero in datetime (1 gennaio dell'anno)
                    df[col] = pd.to_datetime(df[col], format='%Y')
        return df

    @staticmethod
    def _convert_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte colonne numeriche dichiarate come categoriche.

        Alcune colonne nel DB sono codificate come numeriche ma rappresentano
        categorie discrete (es. codici, identificativi). Questa conversione
        evita che vengano trattate come valori continui nei grafici.

        Args:
            df: DataFrame da processare

        Returns:
            DataFrame con colonne categoriche convertite
        """
        for col in df.columns:
            if col in FORCED_CATEGORICAL_COLUMNS:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Converti in stringa prima, poi in category
                    df[col] = df[col].astype(str).astype('category')
        return df

    @staticmethod
    def from_dict_safe(
        result_dict: Dict[str, Any],
        default_on_error: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Versione safe di from_dict che restituisce un DataFrame di default in caso di errore.

        Args:
            result_dict: Dict con struttura tool output
            default_on_error: DataFrame da restituire in caso di errore (default: DataFrame vuoto)

        Returns:
            DataFrame convertito o default_on_error in caso di errore
        """
        try:
            return DataFrameConverter.from_dict(result_dict)
        except (KeyError, ValueError, TypeError) as e:
            if default_on_error is None:
                # Restituisci DataFrame vuoto
                return pd.DataFrame()
            return default_on_error

    @staticmethod
    def validate_structure(result_dict: Dict[str, Any]) -> bool:
        """
        Valida che il dizionario abbia la struttura corretta per la conversione.

        Args:
            result_dict: Dict da validare

        Returns:
            True se la struttura è valida, False altrimenti
        """
        try:
            # Verifica presenza campi
            required_fields = {"columns", "data", "row_count"}
            if not all(field in result_dict for field in required_fields):
                return False

            # Verifica tipi
            if not isinstance(result_dict["columns"], list):
                return False
            if not isinstance(result_dict["data"], list):
                return False
            if not isinstance(result_dict["row_count"], int):
                return False

            # Verifica consistenza
            if len(result_dict["data"]) != result_dict["row_count"]:
                return False

            # Verifica che ogni riga abbia lo stesso numero di colonne
            num_columns = len(result_dict["columns"])
            for row in result_dict["data"]:
                if not isinstance(row, (list, tuple)):
                    return False
                if len(row) != num_columns:
                    return False

            return True

        except Exception:
            return False
