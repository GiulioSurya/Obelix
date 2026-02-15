# sql/main.py - Test Text-to-SQL con framework Obelix (Factory + SharedMemoryGraph)
"""
Flow:
  column_filter  ──▶  sql_agent
        │                     │
        └──────────┬──────────┘
            coordinator (orchestrator)

Il coordinator chiama column_filter per selezionare tabelle/colonne rilevanti,
poi sql_agent riceve il contesto via SharedMemoryGraph e costruisce
lo schema arricchito a runtime prima di generare/eseguire la query SQL.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from sql.agents import SqlAgent, ColumnFilterAgent, CoordinatorAgent
from sql.connections.db_connection.oracle_connection import (
    create_oracle_pool)

from sql.connections.db_connection.postgres_connection import get_postgres_connection
from obelix.core.agent import AgentFactory, SharedMemoryGraph
from obelix.core.agent.shared_memory import PropagationPolicy
from obelix.infrastructure.k8s import YamlConfig
from obelix.infrastructure.logging import setup_logging

# Provider imports
from obelix.adapters.outbound.oci.connection import OCIConnection
from obelix.adapters.outbound.oci.provider import OCILLm

load_dotenv()
setup_logging(console_level="INFO")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_CACHE_PATH = PROJECT_ROOT / "sql" / "database" / "cache" / "schema_cache" / "bilancio_schema.json"

agents_config = YamlConfig(os.getenv("CONFIG_PATH"))

coordinator_config = agents_config.get("agents.coordinator")
column_filter_config = agents_config.get("agents.table_router")
sql_config = agents_config.get("agents.sql_agent")


def create_oci_provider(model_id: str = "openai.gpt-oss-120b", **kwargs) -> OCILLm:
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

    return OCILLm(connection=oci_connection, model_id=model_id, **kwargs)



def create_memory_graph() -> SharedMemoryGraph:
    """column_filter -> sql_agent: le colonne selezionate diventano contesto per la generazione SQL."""
    graph = SharedMemoryGraph()
    graph.add_agent("column_filter")
    graph.add_agent("sql_agent")
    graph.add_edge("column_filter", "sql_agent", policy=PropagationPolicy.LAST_TOOL_RESULT)
    graph.add_edge("sql_agent", "coordinator", policy=PropagationPolicy.LAST_TOOL_RESULT)
    return graph


def create_factory() -> AgentFactory:
    """Configura la factory con tutti gli agent e le dipendenze."""
    oracle_pool = create_oracle_pool()
    postgres_conn = get_postgres_connection()
    memory_graph = create_memory_graph()

    factory = AgentFactory()
    factory.with_memory_graph(memory_graph)


    factory.register(
        name="column_filter",
        cls=ColumnFilterAgent,
        subagent_description=(
            "Analizza la domanda utente e seleziona le tabelle e colonne rilevanti "
            "dallo schema del database. Chiamarlo SEMPRE per primo."
        ),
        defaults={
            "cache_path": SCHEMA_CACHE_PATH,
            "provider": create_oci_provider(**column_filter_config),
        },
    )


    factory.register(
        name="sql_agent",
        cls=SqlAgent,
        subagent_description=(
            "Interroga il database Oracle di bilancio pubblico. "
            "NON scrivere SQL tu: passa la domanda in linguaggio naturale nel campo query "
            "e questo agent genererà ed eseguirà la SQL autonomamente. "
            "Il campo enhanced_query serve per espandere la domanda con sinonimi e termini affini "
            "per migliorare la ricerca semantica nel database. "
        ),
        defaults={
            "postgres_conn": postgres_conn,
            "oracle_pool": oracle_pool,
            "provider": create_oci_provider(**sql_config),
        },
    )


    factory.register(
        name="coordinator",
        cls=CoordinatorAgent,
        defaults={
            "system_message": agents_config.get("prompts.coordinator"),
            "provider": create_oci_provider(**coordinator_config),
        },
    )

    return factory


# ========== MAIN ==========
if __name__ == "__main__":
    factory = create_factory()

    coordinator = factory.create(
        "coordinator",
        subagents=["column_filter", "sql_agent"],
    )

    print("Chat con il Coordinator (digita 'exit' o 'quit' per uscire, 'history' per la conversation history)\n")

    while True:
        try:
            user_input = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChiusura chat.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Chiusura chat.")
            break

        if user_input.lower() == "history":
            print("\n" + "=" * 50)
            print("CONVERSATION HISTORY")
            print("=" * 50)
            for element in coordinator.conversation_history:
                print(element.model_dump_json(indent=4))
            print("=" * 50 + "\n")
            continue

        response = coordinator.execute_query(user_input)
        print(f"\nCoordinator: {response.content}\n")
