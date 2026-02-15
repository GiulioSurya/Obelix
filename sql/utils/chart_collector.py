# src/utils/chart_collector.py
"""
Chart Collector - Session-Isolated via SessionContext
======================================================

Raccoglie i chart HTML generati dai tool senza passarli attraverso i tool results,
risparmiando token nelle conversazioni con l'LLM.

IMPORTANTE: Lo storage è ora gestito da SessionContext.
Ogni richiesta ha il proprio storage isolato automaticamente.

Pattern: Facade over SessionContext for chart management.
"""
from typing import Optional

from obelix.session_context import get_session, get_session_or_none
from obelix.logging_config import get_logger

logger = get_logger(__name__)


class ChartCollector:
    """
    Collector per raccogliere chart HTML generati durante l'esecuzione.

    Facade che delega lo storage a SessionContext, garantendo
    isolamento automatico tra richieste concorrenti.

    Usage:
        # Nel tool
        ChartCollector.add_chart(chart_html, metadata)

        # Nel workflow dopo esecuzione agente
        charts = ChartCollector.get_charts()

        # Cleanup (opzionale con SessionContext)
        ChartCollector.clear()

    Thread Safety:
        - Ogni richiesta async ha il proprio storage isolato via SessionContext
        - Richieste concorrenti NON interferiscono tra loro
    """

    @classmethod
    def add_chart(cls, chart_html: str, metadata: dict) -> None:
        """
        Aggiunge un chart HTML al collector della sessione corrente.

        Args:
            chart_html: HTML del chart generato (Plotly)
            metadata: Metadata del chart (chart_type, data_points, columns_used, etc.)
        """
        session = get_session()
        session.add_chart(chart_html, metadata)
        logger.debug(f"[{session.session_id[:8]}] Chart added, total: {len(session.charts)}")

    @classmethod
    def set_data_table(cls, table_html: str) -> None:
        """
        Imposta la tabella dati master (chiamata una sola volta per sessione).

        Args:
            table_html: HTML della tabella dati completa
        """
        session = get_session()
        if session.data_table is None:
            session.data_table = table_html
            logger.debug(f"[{session.session_id[:8]}] Data table set")

    @classmethod
    def get_charts(cls) -> list[dict]:
        """
        Recupera tutti i chart raccolti per la sessione corrente.

        Returns:
            Lista di dict con struttura: {"html": str, "metadata": dict}
        """
        session = get_session()
        return session.charts.copy()

    @classmethod
    def get_charts_html(cls) -> list[str]:
        """
        Recupera solo gli HTML dei chart (per compatibilità con workflow esistente).

        Returns:
            Lista di stringhe HTML (tabella + chart)
        """
        session = get_session()
        return session.get_charts_html()

    @classmethod
    def get_data_table(cls) -> Optional[str]:
        """
        Recupera la tabella dati master per la sessione corrente.

        Returns:
            HTML della tabella o None se non impostata
        """
        session = get_session()
        return session.data_table

    @classmethod
    def clear(cls) -> None:
        """
        Pulisce i chart per la sessione corrente.

        Nota: Con SessionContext, il cleanup completo avviene automaticamente
        alla fine del session_scope(). Questo metodo pulisce solo i chart,
        non l'intera sessione.
        """
        session = get_session_or_none()
        if session is not None:
            session.clear_charts()
            logger.debug(f"[{session.session_id[:8]}] Charts cleared")

    @classmethod
    def get_count(cls) -> int:
        """
        Restituisce il numero di chart raccolti per la sessione corrente.

        Returns:
            Numero di chart nel collector
        """
        session = get_session()
        return len(session.charts)
