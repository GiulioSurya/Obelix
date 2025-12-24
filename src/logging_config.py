# src/logging_config.py
"""
Modulo di configurazione logging per SophIA.

Usa Loguru come backend. Questo modulo fornisce due funzioni principali:
- setup_logging(): configura il logger all'avvio dell'applicazione
- get_logger(name): ottiene un logger "bindato" con il nome del modulo

=============================================================================
LOGURU - CONCETTI BASE
=============================================================================

Loguru ha un UNICO logger globale chiamato `logger`. Non crei istanze multiple
come con il logging standard. Invece, usi `bind()` per aggiungere contesto.

Esempio concettuale:
    from loguru import logger

    # Questo e' SEMPRE lo stesso logger globale
    logger.info("messaggio")

    # bind() aggiunge "extra" al contesto, ma e' ancora lo stesso logger
    logger_modulo = logger.bind(name="mio_modulo")
    logger_modulo.info("messaggio")  # Ora include name="mio_modulo"

=============================================================================
LIVELLI DI LOG (dal meno al piu' grave)
=============================================================================

1. TRACE (5)    - Dettagli estremi, per debug profondo
                  Esempio: "Entrando in funzione X con param Y"

2. DEBUG (10)   - Info utili durante sviluppo
                  Esempio: "Query SQL generata: SELECT..."

3. INFO (20)    - Eventi normali dell'applicazione
                  Esempio: "Agent SQLGenerator inizializzato"

4. SUCCESS (25) - Operazioni completate con successo (specifico Loguru)
                  Esempio: "Query eseguita, 150 righe restituite"

5. WARNING (30) - Situazioni anomale ma gestite
                  Esempio: "Timeout LLM, riprovo..."

6. ERROR (40)   - Errori che impediscono un'operazione
                  Esempio: "Impossibile connettersi al DB"

7. CRITICAL (50)- Errori fatali, applicazione non puo' continuare
                  Esempio: "File configurazione mancante"

=============================================================================
QUANDO USARE OGNI LIVELLO
=============================================================================

DEBUG:
    - Valori di variabili durante esecuzione
    - Query SQL prima dell'esecuzione
    - Payload di richieste/risposte API
    - Stato interno degli oggetti

    logger.debug(f"Tool call ricevuta: {tool_call.name}")
    logger.debug(f"Parametri: {tool_call.arguments}")

INFO:
    - Avvio/stop di componenti
    - Operazioni di business completate
    - Cambi di stato significativi

    logger.info(f"Agent {self.agent_name} inizializzato")
    logger.info("Pipeline esecuzione avviata")

WARNING:
    - Retry automatici
    - Configurazioni mancanti con default usato
    - Rate limiting applicato
    - Deprecation notices

    logger.warning(f"Tentativo {attempt}/3 fallito, riprovo...")
    logger.warning("Parametro X non specificato, uso default Y")

ERROR:
    - Eccezioni catturate e gestite
    - Operazioni fallite (ma app continua)
    - Risorse non disponibili

    logger.error(f"Tool {tool_name} non trovato: {e}")
    logger.error("Connessione DB fallita", exc_info=True)

CRITICAL:
    - Errori che richiedono shutdown
    - Corruzioni di stato irrecuperabili
    - Violazioni di sicurezza

    logger.critical("Impossibile inizializzare provider LLM")

=============================================================================
CONFIGURAZIONE HANDLER (add)
=============================================================================

Loguru parte con un handler di default (stderr). Per personalizzare:

    logger.remove()  # Rimuove handler di default

    # Aggiungi handler personalizzato
    logger.add(
        sink,           # Dove scrivere: file path, sys.stderr, funzione custom
        level,          # Livello minimo: "DEBUG", "INFO", etc.
        format,         # Formato messaggio
        rotation,       # Quando ruotare file: "50 MB", "1 day", "12:00"
        retention,      # Quanto tenere vecchi file: "7 days", "1 week"
        colorize,       # True/False per colori (solo console)
        serialize,      # True per output JSON
        filter,         # Funzione per filtrare messaggi
    )

Il metodo add() ritorna un ID che puoi usare per rimuovere l'handler:

    handler_id = logger.add("file.log")
    logger.remove(handler_id)  # Rimuove solo questo handler

=============================================================================
"""

from loguru import logger
from pathlib import Path
import sys


# Flag per evitare setup multipli
_is_configured = False


def setup_logging(
    level: str = "INFO",
    console_level: str = "DEBUG",
    log_dir: str = "logs",
    log_filename: str = "sophia.log"
) -> None:
    """
    Configura il logging per l'applicazione.

    Chiama questa funzione UNA VOLTA all'avvio dell'app (es. in main.py).
    Chiamate successive vengono ignorate.

    Args:
        level: Livello minimo per FILE. Default: "DEBUG" (cattura tutto)
        console_level: Livello minimo per CONSOLE. Default: "INFO"
                       Valori: "TRACE", "DEBUG", "INFO", "SUCCESS",
                       "WARNING", "ERROR", "CRITICAL"
        log_dir: Directory per i file di log. Viene creata se non esiste.
                 Default: "logs"
        log_filename: Nome del file di log. Default: "sophia.log"

    Esempio:
        # In main.py
        from src.logging_config import setup_logging

        setup_logging()  # Default: file=DEBUG, console=INFO

        # Solo errori in console, tutto su file
        setup_logging(console_level="WARNING")

        # Debug anche in console
        setup_logging(console_level="DEBUG")

    Comportamento:
        - Rimuove handler di default di Loguru
        - Aggiunge handler FILE (level=DEBUG, con rotazione 50 MB)
        - Aggiunge handler CONSOLE (level=INFO, con colori)
    """
    global _is_configured

    if _is_configured:
        return

    # Crea directory log se non esiste
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Rimuovi handler di default (stderr)
    logger.remove()

    # Formato log: timestamp | livello | modulo:funzione:linea | messaggio
    # {name} viene dal bind() che facciamo in get_logger()
    log_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level:<8} | "
        "{extra[name]}:{function}:{line} | "
        "{message}"
    )

    # Handler file con rotazione
    logger.add(
        sink=log_path / log_filename,  # Path del file
        level=level,                    # Livello minimo (DEBUG)
        format=log_format,              # Formato definito sopra
        rotation="50 MB",               # Ruota quando file supera 50 MB
        retention="7 days",             # Mantieni file per 7 giorni
        encoding="utf-8",               # Encoding file
    )

    # Formato console: piu' compatto, senza timestamp completo
    console_format = (
        "<level>{level:<8}</level> | "
        "<cyan>{extra[name]}</cyan>:<cyan>{function}</cyan> | "
        "{message}"
    )

    # Handler console con colori
    logger.add(
        sink=sys.stderr,                # Output: console (stderr)
        level=console_level,            # Livello minimo (INFO di default)
        format=console_format,          # Formato compatto
        colorize=True,                  # Colori attivi
    )

    _is_configured = True

    # Log iniziale per confermare setup
    logger.bind(name="logging_config").info(
        f"Logging configurato - file={level}, console={console_level}, path={log_path / log_filename}"
    )


def get_logger(name: str):
    """
    Ottiene un logger con il nome del modulo bindato.

    Il nome viene incluso in ogni messaggio di log, permettendo di
    identificare da quale modulo proviene il log.

    Args:
        name: Nome del modulo. Usa sempre __name__ per coerenza.

    Returns:
        Logger Loguru con il nome bindato.

    Esempio:
        # All'inizio del modulo
        from src.logging_config import get_logger

        logger = get_logger(__name__)

        # Poi nel codice
        logger.debug("Questo e' un messaggio debug")
        logger.info("Operazione completata")
        logger.warning("Attenzione: valore inatteso")
        logger.error("Errore durante esecuzione")

    Note:
        - __name__ restituisce il path del modulo (es. "src.base_agent.base_agent")
        - Il logger ritornato e' sempre lo stesso logger globale Loguru,
          ma con contesto aggiuntivo (il nome)
        - Se setup_logging() non e' stato chiamato, il comportamento
          e' quello di default Loguru (stderr)
    """
    return logger.bind(name=name)


# =============================================================================
# ESEMPI D'USO RAPIDI
# =============================================================================
#
# --- In main.py (una volta sola) ---
#
# from src.logging_config import setup_logging
# setup_logging()
#
# --- In qualsiasi altro modulo ---
#
# from src.logging_config import get_logger
# logger = get_logger(__name__)
#
# def mia_funzione():
#     logger.debug("Inizio funzione")
#     try:
#         # ... codice ...
#         logger.info("Operazione completata")
#     except Exception as e:
#         logger.error(f"Errore: {e}")
#         raise
#
# --- Log con variabili (f-string o format) ---
#
# logger.debug(f"Valore x={x}, y={y}")
# logger.info("User {} logged in", username)  # Lazy formatting
#
# --- Log con eccezione (traceback completo) ---
#
# try:
#     risky_operation()
# except Exception:
#     logger.exception("Operazione fallita")  # Include traceback
#
# --- Log con dati strutturati ---
#
# logger.info("Request received", extra={"method": "POST", "path": "/api"})
#
