"""
Year Detector Module
====================

Rileva se una colonna numerica rappresenta anni usando euristiche a strati:
1. Regola "hard": range temporale plausibile [1900, anno_corrente + 2]
2. Esclusioni: valori speciali (0, 9999, etc.)
3. Analisi distribuzione: concentrazione valori e prossimità al presente
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Set

import pandas as pd


@dataclass
class YearDetectionResult:
    """Risultato della detection di una colonna anno."""
    is_year: bool
    confidence: float  # 0.0 - 1.0
    reason: str


class YearDetector:
    """
    Rileva se una colonna numerica rappresenta anni.

    Usa euristiche a strati:
    1. Hard rule: range [min_year, max_year]
    2. Esclusioni: valori speciali
    3. Distribuzione colonna: % valori in range + concentrazione vicino al presente

    Example:
        >>> detector = YearDetector()
        >>> result = detector.detect(pd.Series([2020, 2021, 2022, 2023]), "ANNO")
        >>> result.is_year
        True
        >>> result.confidence
        1.0
    """

    # Pattern nei nomi colonna che suggeriscono un anno
    YEAR_NAME_PATTERNS = frozenset([
        'ANNO', 'YEAR', 'ESERCIZIO', 'ANNUALITA', 'FISCAL_YEAR',
        'ANNO_RIF', 'ANNO_RIFERIMENTO', 'YR'
    ])

    # Pattern nei nomi colonna che suggeriscono un codice (NON anno)
    CODE_NAME_PATTERNS = frozenset([
        'COD', 'CODE', 'CODICE', 'ID', 'MATRICOLA', 'NUMERO',
        'NUM', 'CAP', 'ZIP', 'PIN', 'SKU'
    ])

    def __init__(
        self,
        min_year: int = 1900,
        max_year: Optional[int] = None,
        excluded_values: Optional[Set[int]] = None,
        min_valid_ratio: float = 0.9,
        recency_boost_years: int = 10
    ):
        """
        Args:
            min_year: Anno minimo plausibile (default: 1900)
            max_year: Anno massimo plausibile (default: anno_corrente + 2)
            excluded_values: Valori da escludere (default: {0, 9999})
            min_valid_ratio: Frazione minima di valori in range per considerare anno (default: 0.9)
            recency_boost_years: Finestra anni recenti per boost confidenza (default: 10)
        """
        self.min_year = min_year
        self.max_year = max_year or (datetime.now().year + 2)
        self.excluded_values = excluded_values or {0, 9999}
        self.min_valid_ratio = min_valid_ratio
        self.recency_boost_years = recency_boost_years
        self._current_year = datetime.now().year

    def detect(self, series: pd.Series, col_name: Optional[str] = None) -> YearDetectionResult:
        """
        Rileva se una Series rappresenta anni.

        Args:
            series: Colonna pandas da analizzare
            col_name: Nome colonna (opzionale, usato per euristiche basate sul nome)

        Returns:
            YearDetectionResult con is_year, confidence e reason
        """
        # Shortcut: se già datetime, non serve inferenza
        if pd.api.types.is_datetime64_any_dtype(series):
            return YearDetectionResult(
                is_year=False,
                confidence=1.0,
                reason="Colonna già datetime"
            )

        # Deve essere numerica
        if not pd.api.types.is_numeric_dtype(series):
            return YearDetectionResult(
                is_year=False,
                confidence=1.0,
                reason="Colonna non numerica"
            )

        # Rimuovi null
        values = series.dropna()

        if len(values) == 0:
            return YearDetectionResult(
                is_year=False,
                confidence=1.0,
                reason="Colonna vuota"
            )

        # Check nome colonna per esclusione rapida (codici)
        if col_name and self._looks_like_code_column(col_name):
            return YearDetectionResult(
                is_year=False,
                confidence=0.8,
                reason=f"Nome colonna suggerisce codice: {col_name}"
            )

        # --- REGOLA 1: Hard rule - range plausibile ---
        # Verifica che siano interi (o float senza decimali)
        if not self._are_integer_like(values):
            return YearDetectionResult(
                is_year=False,
                confidence=0.9,
                reason="Valori con decimali, non possono essere anni"
            )

        int_values = values.astype(int)

        # Escludi valori speciali
        filtered_values = int_values[~int_values.isin(self.excluded_values)]

        if len(filtered_values) == 0:
            return YearDetectionResult(
                is_year=False,
                confidence=0.7,
                reason="Tutti i valori sono esclusi (valori speciali)"
            )

        # --- REGOLA 2: Check range ---
        in_range_mask = (filtered_values >= self.min_year) & (filtered_values <= self.max_year)
        valid_ratio = in_range_mask.sum() / len(filtered_values)

        if valid_ratio < self.min_valid_ratio:
            return YearDetectionResult(
                is_year=False,
                confidence=0.9,
                reason=f"Solo {valid_ratio:.1%} valori in range [{self.min_year}, {self.max_year}]"
            )

        # --- REGOLA 3: Analisi distribuzione ---
        confidence = self._calculate_confidence(filtered_values, col_name)

        # Soglia decisionale
        is_year = confidence >= 0.6

        if is_year:
            reason = self._build_positive_reason(filtered_values, col_name, confidence)
        else:
            reason = f"Confidenza insufficiente: {confidence:.2f}"

        return YearDetectionResult(
            is_year=is_year,
            confidence=confidence,
            reason=reason
        )

    def _looks_like_code_column(self, col_name: str) -> bool:
        """Check se il nome colonna suggerisce un codice."""
        col_upper = col_name.upper()
        return any(pattern in col_upper for pattern in self.CODE_NAME_PATTERNS)

    def _looks_like_year_column(self, col_name: str) -> bool:
        """Check se il nome colonna suggerisce un anno."""
        col_upper = col_name.upper()
        return any(pattern in col_upper for pattern in self.YEAR_NAME_PATTERNS)

    def _are_integer_like(self, values: pd.Series) -> bool:
        """Check se i valori sono interi (o float senza parte decimale)."""
        try:
            return (values == values.astype(int)).all()
        except (ValueError, OverflowError):
            return False

    def _calculate_confidence(self, values: pd.Series, col_name: Optional[str]) -> float:
        """
        Calcola confidenza basata su distribuzione e nome colonna.

        Fattori:
        - Nome colonna contiene pattern anno: +0.3
        - Valori concentrati vicino al presente: +0.2
        - Pochi valori unici (tipico di anni): +0.1
        - Range ristretto (es. 2020-2025): +0.1
        """
        confidence = 0.5  # Base

        # Boost da nome colonna
        if col_name and self._looks_like_year_column(col_name):
            confidence += 0.3

        # Boost da prossimità al presente
        recency_start = self._current_year - self.recency_boost_years
        recent_ratio = ((values >= recency_start) & (values <= self._current_year + 2)).sum() / len(values)
        if recent_ratio >= 0.5:
            confidence += 0.2 * recent_ratio  # Proporzionale

        # Boost da pochi valori unici (tipico di colonne anno)
        unique_ratio = values.nunique() / len(values)
        if unique_ratio < 0.1:  # Meno del 10% valori unici
            confidence += 0.1

        # Boost da range ristretto
        value_range = values.max() - values.min()
        if value_range <= 20:  # Range di max 20 anni
            confidence += 0.1

        return min(confidence, 1.0)

    def _build_positive_reason(
        self,
        values: pd.Series,
        col_name: Optional[str],
        confidence: float
    ) -> str:
        """Costruisce una reason descrittiva per detection positiva."""
        parts = [f"Confidenza: {confidence:.2f}"]

        if col_name and self._looks_like_year_column(col_name):
            parts.append(f"nome colonna '{col_name}' suggerisce anno")

        parts.append(f"range [{int(values.min())}, {int(values.max())}]")
        parts.append(f"{values.nunique()} valori unici su {len(values)} righe")

        return ", ".join(parts)
