# src/session_context.py
"""
Session Context - Request-Isolated State Management
====================================================

Gestisce lo stato isolato per ogni richiesta/sessione utente.
Usa ContextVar per garantire isolamento tra richieste concorrenti.

Pattern: Context Manager + ContextVar per isolamento async-safe.

Usage:
    from src.session_context import session_scope, get_session

    # Entry point (api.py, main.py)
    async with session_scope() as session:
        # Tutto il codice qui vede la stessa sessione isolata
        result = await execute_workflow(query)

    # Ovunque nel codice (tools, collectors, etc.)
    session = get_session()
    session.charts.append(chart)
    print(session.session_id)

Thread Safety:
    ContextVar garantisce che ogni "contesto async" (richiesta) abbia
    il proprio SessionContext isolato. Richieste concorrenti NON
    interferiscono tra loro.
"""
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, Any
from uuid import uuid4

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SessionContext:
    """
    Contesto isolato per una singola sessione/richiesta.

    Contiene tutto lo stato che deve essere isolato tra richieste:
    - session_id: identificatore univoco della sessione
    - charts: lista dei chart generati (per ChartCollector)
    - data_table: tabella dati HTML (per ChartCollector)
    - metadata: dizionario per dati aggiuntivi estensibili

    Attributes:
        session_id: Identificatore univoco della sessione (UUID)
        charts: Lista di chart generati durante il workflow
        data_table: HTML della tabella dati (opzionale)
        metadata: Dati aggiuntivi della sessione (estensibile)
    """
    session_id: str
    charts: list[dict] = field(default_factory=list)
    data_table: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_chart(self, chart_html: str, chart_metadata: dict) -> None:
        """
        Aggiunge un chart alla sessione.

        Args:
            chart_html: HTML del chart (Plotly)
            chart_metadata: Metadata del chart (tipo, colonne, etc.)
        """
        self.charts.append({
            "html": chart_html,
            "metadata": chart_metadata
        })

    def get_charts_html(self) -> list[str]:
        """
        Restituisce gli HTML dei chart (tabella + grafici).

        Returns:
            Lista di stringhe HTML
        """
        html_list = []
        if self.data_table:
            html_list.append(self.data_table)
        for chart in self.charts:
            html_list.append(chart["html"])
        return html_list

    def clear_charts(self) -> None:
        """Pulisce i chart della sessione."""
        self.charts.clear()
        self.data_table = None


# ContextVar per lo storage della sessione corrente
_current_session: ContextVar[Optional[SessionContext]] = ContextVar(
    'current_session',
    default=None
)


def init_session(session_id: str = None) -> SessionContext:
    """
    Inizializza una nuova sessione per il contesto corrente.

    Args:
        session_id: ID sessione (opzionale, genera UUID se non fornito)

    Returns:
        SessionContext appena creato

    Raises:
        RuntimeError: Se una sessione è già attiva nel contesto corrente
    """
    existing = _current_session.get()
    if existing is not None:
        raise RuntimeError(
            f"Sessione già attiva: {existing.session_id}. "
            "Chiamare clear_session() prima di inizializzarne una nuova."
        )

    # TODO: session_id attualmente generato come UUID casuale se non fornito.
    #       Quando verrà implementato il login, gli entry point (api.py) passeranno
    #       user_id dall'autenticazione invece di lasciare generare UUID.
    if session_id is None:
        session_id = str(uuid4())

    session = SessionContext(session_id=session_id)
    _current_session.set(session)

    logger.debug(f"Session initialized: {session_id}")
    return session


def get_session() -> SessionContext:
    """
    Ottiene la sessione corrente.

    Returns:
        SessionContext attivo per questo contesto

    Raises:
        RuntimeError: Se nessuna sessione è attiva
    """
    session = _current_session.get()
    if session is None:
        raise RuntimeError(
            "Nessuna sessione attiva. "
            "Assicurarsi di essere dentro un 'async with session_scope()' "
            "o di aver chiamato init_session()."
        )
    return session


def get_session_or_none() -> Optional[SessionContext]:
    """
    Ottiene la sessione corrente o None se non attiva.

    Utile per codice che può funzionare sia con che senza sessione.

    Returns:
        SessionContext attivo o None
    """
    return _current_session.get()


def clear_session() -> None:
    """
    Pulisce la sessione corrente.

    Sicuro da chiamare anche se nessuna sessione è attiva.
    """
    session = _current_session.get()
    if session is not None:
        logger.debug(f"Session cleared: {session.session_id}")
    _current_session.set(None)


@asynccontextmanager
async def session_scope(session_id: str = None):
    """
    Context manager async per gestione automatica della sessione.

    Inizializza la sessione all'ingresso e garantisce cleanup all'uscita,
    anche in caso di eccezioni.

    Args:
        session_id: ID sessione (opzionale, genera UUID se non fornito)

    Yields:
        SessionContext attivo per questo scope

    Usage:
        async with session_scope() as session:
            print(session.session_id)
            # ... tutto il codice qui vede questa sessione
        # Cleanup automatico

        # Con session_id esplicito (es. da WebSocket)
        async with session_scope(session_id="abc-123") as session:
            # ...

    Example:
        # In api.py
        @app.post("/query")
        async def query_endpoint(request: QueryRequest):
            async with session_scope() as session:
                result = await execute_text_to_sql_workflow(request.query)
                return result

        # In WebSocket
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            session_id = str(uuid4())
            async with session_scope(session_id) as session:
                bridge.register_websocket(session.session_id, websocket)
                # ...
    """
    session = init_session(session_id)
    logger.info(f"Session scope started: {session.session_id}")
    try:
        yield session
    finally:
        logger.info(f"Session scope ended: {session.session_id}")
        clear_session()
