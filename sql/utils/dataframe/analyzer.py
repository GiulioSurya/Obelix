"""
DataFrame Analyzer Module
=========================

Analizza DataFrame pandas ed estrae metadata ricchi per agent LLM.
Non modifica i dati, solo li analizza per fornire informazioni strutturate.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any

from sql.utils.dataframe.year_detector import YearDetector


class DataFrameAnalyzer:
    """
    Analizza un DataFrame e produce metadata ricchi per un LLM agent.
    Non trasforma i dati, solo li analizza.
    """

    # Detector condiviso per rilevare colonne anno
    _year_detector = YearDetector()

    @staticmethod
    def _load_column_descriptions() -> Dict[str, str]:
        """
        Carica le descrizioni delle colonne dallo schema JSON.
        Per colonne duplicate in più tabelle, prende la prima descrizione trovata.

        Returns:
            Dizionario {column_name: description}
        """
        # Path dello schema JSON (relativo alla root del progetto)
        schema_path = Path(__file__).parent.parent.parent / "database" / "cache" / "schema_cache" / "bilancio_schema.json"

        if not schema_path.exists():
            return {}

        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)

            column_descriptions = {}

            # Itera su tutte le tabelle nello schema
            for table in schema.get("tables", []):
                for column in table.get("columns", []):
                    col_name = column.get("column_name")
                    col_desc = column.get("description")

                    # Prendi solo la prima descrizione (ignora duplicati)
                    if col_name and col_desc and col_name not in column_descriptions:
                        column_descriptions[col_name] = col_desc

            return column_descriptions

        except Exception as e:
            print(f"Warning: Failed to load column descriptions from schema: {e}")
            return {}

    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analizza DataFrame e produce metadata ricchi.
        Include descrizioni delle colonne dallo schema database.

        Args:
            df: DataFrame da analizzare

        Returns:
            Dizionario con metadata completi
        """
        metadata = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": []
        }

        # Carica descrizioni colonne dallo schema
        column_descriptions = DataFrameAnalyzer._load_column_descriptions()

        # Analizza ogni colonna
        for col in df.columns:
            col_meta = DataFrameAnalyzer._analyze_column(
                df[col],
                col
            )

            # Aggiungi descrizione se disponibile
            if col in column_descriptions:
                col_meta["description"] = column_descriptions[col]

            metadata["columns"].append(col_meta)

        # Analisi relazioni tra colonne e suggerimenti
        metadata["insights"] = DataFrameAnalyzer._generate_insights(df)

        return metadata

    @staticmethod
    def _analyze_column(series: pd.Series, col_name: str) -> Dict[str, Any]:
        """Analizza una singola colonna in dettaglio"""
        col_info = {
            "name": col_name,
            "semantic_type": None
        }

        # Determina tipo semantico e statistiche
        if pd.api.types.is_bool_dtype(series):
            col_info["semantic_type"] = "boolean"
            if series.notna().any():
                value_counts = series.value_counts()
                col_info["stats"] = {
                    "true_count": int(value_counts.get(True, 0)),
                    "false_count": int(value_counts.get(False, 0))
                }

        elif pd.api.types.is_numeric_dtype(series):
            # Prima check se è una colonna anno
            year_result = DataFrameAnalyzer._year_detector.detect(series, col_name)

            if year_result.is_year:
                col_info["semantic_type"] = "temporal"
                col_info["unique_values"] = int(series.nunique())
                col_info["granularity"] = "yearly"
                col_info["year_detection"] = {
                    "confidence": year_result.confidence,
                    "reason": year_result.reason
                }
                # Aggiungi date_range per coerenza con colonne datetime
                if series.notna().any():
                    col_info["date_range"] = {
                        "start": str(int(series.min())),
                        "end": str(int(series.max()))
                    }
            else:
                col_info["semantic_type"] = "numeric"
                col_info["unique_values"] = int(series.nunique())
                if series.notna().any():
                    col_info["stats"] = {
                        "min": round(float(series.min()), 2),
                        "q1": round(float(series.quantile(0.25)), 2),
                        "median": round(float(series.median()), 2),
                        "q3": round(float(series.quantile(0.75)), 2),
                        "max": round(float(series.max()), 2),
                        "mean": round(float(series.mean()), 2)
                    }

        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info["semantic_type"] = "temporal"
            col_info["unique_values"] = int(series.nunique())
            if series.notna().any():
                min_date = series.min()
                max_date = series.max()
                col_info["date_range"] = {
                    "start": str(min_date),
                    "end": str(max_date)
                }

                # Rileva granularita (daily, monthly, yearly)
                if series.notna().sum() > 1:
                    time_diffs = series.dropna().sort_values().diff().dropna()
                    if len(time_diffs) > 0:
                        median_diff = time_diffs.median()
                        if median_diff < pd.Timedelta(days=2):
                            col_info["granularity"] = "daily"
                        elif median_diff < pd.Timedelta(days=40):
                            col_info["granularity"] = "monthly"
                        else:
                            col_info["granularity"] = "yearly"

        else:  # object/string
            unique_count = series.nunique()
            total_count = len(series)
            uniqueness_ratio = unique_count / total_count if total_count > 0 else 0

            if uniqueness_ratio < 0.5 and unique_count < 50:
                col_info["semantic_type"] = "categorical"
                col_info["unique_values"] = unique_count
                # Top 10 categorie
                value_counts = series.value_counts().head(10)
                col_info["top_categories"] = [str(val) for val in value_counts.index]
            else:
                col_info["semantic_type"] = "text"
                col_info["unique_values"] = unique_count

        return col_info

    @staticmethod
    def _generate_insights(df: pd.DataFrame) -> Dict[str, Any]:
        """Genera insights automatici sui dati per aiutare l'agent"""
        insights = {}

        # Conta tipi di colonne
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        temporal_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
        categorical_cols = []

        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < 50:
                categorical_cols.append(col)

        insights["summary"] = {
            "numeric": numeric_cols,
            "temporal": temporal_cols,
            "categorical": categorical_cols
        }

        return insights