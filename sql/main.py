# sql/main.py - Test Text-to-SQL con framework Obelix (Factory + SharedMemoryGraph)
"""
Flow:
  column_filter  ──▶  query_enhancement
        │                     │
        └──────────┬──────────┘
            coordinator (orchestrator)

Il coordinator chiama column_filter per selezionare tabelle/colonne rilevanti,
poi query_enhancement riceve il contesto via SharedMemoryGraph e costruisce
lo schema arricchito a runtime prima di generare/eseguire la query SQL.
"""
import os
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from sql.agents import QueryEnhancementAgent, ColumnFilterAgent, CoordinatorAgent
from sql.connections.db_connection.oracle_connection import (
    create_oracle_pool,
    close_oracle_pool,
)
from sql.connections.db_connection.postgres_connection import get_postgres_connection
from src.core.agent import AgentFactory, SharedMemoryGraph
from src.core.agent.shared_memory import PropagationPolicy
from src.infrastructure.k8s import YamlConfig
from src.infrastructure.logging import setup_logging

# Provider imports
from src.adapters.outbound.oci.connection import OCIConnection
from src.adapters.outbound.oci.provider import OCILLm

load_dotenv()
setup_logging(console_level="TRACE")

# ========== PATHS ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_CACHE_PATH = PROJECT_ROOT / "sql" / "database" / "cache" / "schema_cache" / "bilancio_schema.json"


# ========== STRUCTURED OUTPUT SCHEMA ==========
class ColumnFilterOutput(BaseModel):
    """Schema per la risposta strutturata del ColumnFilterAgent.
    Formato atteso da SemanticSchemaBuilder.enrich_schema(filter=...)."""
    VISTA_BILANCIO_ENTRATA_AI: Optional[List[str]] = Field(
        default=None,
        description="Colonne selezionate dalla vista bilancio entrata",
    )
    VISTA_BILANCIO_SPESA_AI: Optional[List[str]] = Field(
        default=None,
        description="Colonne selezionate dalla vista bilancio spesa",
    )


# ========== PROVIDER SETUP ==========
def create_oci_provider(model_id: str = "openai.gpt-oss-120b") -> OCILLm:
    """Crea un provider OCI con configurazione da infrastructure.yaml."""
    infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
    oci_cfg = infra_config.get("llm_providers.oci")

    oci_connection = OCIConnection({
        "user": oci_cfg["user_id"],
        "fingerprint": oci_cfg["fingerprint"],
        "key_content": oci_cfg["private_key_content"],
        "tenancy": oci_cfg["tenancy"],
        "region": oci_cfg["region"],
    })

    return OCILLm(connection=oci_connection, model_id=model_id)


# ========== SHARED MEMORY GRAPH ==========
def create_memory_graph() -> SharedMemoryGraph:
    """column_filter -> query_enhancement: le colonne selezionate diventano contesto per la generazione SQL."""
    graph = SharedMemoryGraph()
    graph.add_agent("column_filter")
    graph.add_agent("query_enhancement")
    graph.add_edge("column_filter", "query_enhancement", policy=PropagationPolicy.FINAL_RESPONSE_ONLY)
    return graph


# ========== FACTORY SETUP ==========
def create_factory() -> AgentFactory:
    """Configura la factory con tutti gli agent e le dipendenze."""
    provider = create_oci_provider()
    oracle_pool = create_oracle_pool()
    postgres_conn = get_postgres_connection()
    memory_graph = create_memory_graph()

    factory = AgentFactory()
    factory.with_memory_graph(memory_graph)

    # --- Column Filter Agent ---
    factory.register(
        name="column_filter",
        cls=ColumnFilterAgent,
        subagent_description=(
            "Analizza la domanda utente e seleziona le tabelle e colonne rilevanti "
            "dallo schema del database. Chiamarlo SEMPRE per primo."
        ),
        defaults={
            "cache_path": SCHEMA_CACHE_PATH,
            "provider": provider,
            "response_schema": ColumnFilterOutput,
        },
    )

    # --- Query Enhancement Agent ---
    factory.register(
        name="query_enhancement",
        cls=QueryEnhancementAgent,
        subagent_description=("""
        Genera ed esegue query SQL Oracle ottimizzate. 
        Chiamarlo DOPO column_filter, riceve automaticamente il contesto delle colonne selezionate. 
        Accetta enhanced_query: una riformulazione semantica della domanda utente 
        quando lo interroghi formula la query in linguaggio naturale
        con sinonimi e termini affini per migliorare la ricerca nel database."""
        ),
        defaults={
            "postgres_conn": postgres_conn,
            "oracle_pool": oracle_pool,
            "provider": provider,
        },
    )

    # --- Coordinator Agent ---
    # TODO: system_message del coordinator da aggiungere (config o inline)
    factory.register(
        name="coordinator",
        cls=CoordinatorAgent,
        defaults={
            "system_message": (
                "Sei un coordinatore per query su database di bilancio.\n"
                "Hai a disposizione due agent:\n"
                "1. column_filter: seleziona tabelle e colonne rilevanti\n"
                "2. query_enhancement: genera ed esegue la query SQL\n\n"
                "REGOLE:\n"
                "- Chiama SEMPRE column_filter per primo\n"
                "- Poi chiama query_enhancement passando in enhanced_query una riformulazione "
                "semantica della domanda: espandi con sinonimi e termini affini, "
                "elimina parole superflue, mantieni una frase densa e scorrevole\n"
                "- Presenta i risultati in modo chiaro all'utente"
            ),
            "provider": provider,
        },
    )

    return factory


# ========== MAIN ==========
if __name__ == "__main__":
    factory = create_factory()

    coordinator = factory.create(
        "coordinator",
        subagents=["column_filter", "query_enhancement"],
    )

    response = coordinator.execute_query(
        "Quali sono le voci di spesa della missione 03 per il 2025?"
    )

    print("\n" + "=" * 50)
    print("CONVERSATION HISTORY")
    print("=" * 50)
    for element in coordinator.conversation_history:
        print(element.model_dump_json(indent=4))