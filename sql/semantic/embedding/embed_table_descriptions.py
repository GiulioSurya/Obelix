"""
Modulo per embedding delle descrizioni tabelle Oracle.

Crea e popola la tabella `table_descriptions` in PostgreSQL per il routing
semantico delle query verso le tabelle corrette.

Struttura tabella:
- table_name: nome tabella Oracle (VARCHAR, PRIMARY KEY)
- description: descrizione testuale (TEXT)
- embedding: vettore 1024-dim (vector)

Usage:
    python -m src.semantic.embedding.embed_table_descriptions
"""

import json
from pathlib import Path
from typing import Optional

from sql.connections.db_connection.postgres_connection import PostgresConnection, get_postgres_connection
from src.adapters.outbound.embedding.oci_embedding import OCIEmbeddingProvider
from sql.semantic.analysis.query_intent_extractor import QueryIntentExtractor, IntentExtractionConfig


# Nome tabella PostgreSQL per le descrizioni
TABLE_NAME = "table_descriptions"
EMBEDDING_DIM = 1024


class TableDescriptionsManager:
    """
    Gestisce la creazione e popolamento della tabella `table_descriptions`
    per semantic routing delle query.
    """

    def __init__(
        self,
        postgres_conn: Optional[PostgresConnection] = None,
        embedding_provider: Optional[OCIEmbeddingProvider] = None,
        schema_json_path: Optional[str] = None,
        use_intent_extractor: bool = True
    ):
        """
        Args:
            postgres_conn: Connessione PostgreSQL
            embedding_provider: Provider per generare embeddings
            schema_json_path: Path al file JSON con schema (default: bilancio_schema.json)
            use_intent_extractor: Se True, pulisce le descrizioni con QueryIntentExtractor prima dell'embedding
        """
        self.pg_conn = postgres_conn or get_postgres_connection()
        self._embedding_provider = embedding_provider
        self.use_intent_extractor = use_intent_extractor
        self._intent_extractor: Optional[QueryIntentExtractor] = None

        if schema_json_path is None:
            schema_json_path = (
                Path(__file__).parent.parent.parent
                / "database" / "cache" / "schema_cache" / "bilancio_schema.json"
            )

        self.schema_json_path = Path(schema_json_path)
        self.schema_data = self._load_schema()

    @property
    def embedding_provider(self) -> OCIEmbeddingProvider:
        """Lazy initialization dell'embedding provider."""
        if self._embedding_provider is None:
            print("Inizializzando OCI Embedding Provider...")
            self._embedding_provider = OCIEmbeddingProvider(input_type="search_document")
        return self._embedding_provider

    @property
    def intent_extractor(self) -> QueryIntentExtractor:
        """Lazy initialization dell'intent extractor."""
        if self._intent_extractor is None:
            print("Inizializzando QueryIntentExtractor...")
            # Config per descrizioni tabelle: NO default custom stopwords
            config = IntentExtractionConfig(
                use_default_custom_stopwords=False,
                enable_noun_chunks=False,
                enable_lemmatization=True
            )
            self._intent_extractor = QueryIntentExtractor(config=config)
        return self._intent_extractor

    def _load_schema(self) -> dict:
        """Carica lo schema JSON."""
        if not self.schema_json_path.exists():
            raise FileNotFoundError(f"Schema non trovato: {self.schema_json_path}")

        with open(self.schema_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_table(self) -> None:
        """
        Crea la tabella `table_descriptions` con supporto vector.
        """
        print(f"Creando tabella {TABLE_NAME}...")

        # Abilita estensione pgvector
        self.pg_conn.enable_extension("vector")

        # Crea tabella
        self.pg_conn.create_table(
            TABLE_NAME,
            {
                "table_name": "VARCHAR(128) PRIMARY KEY",
                "description": "TEXT NOT NULL",
                "embedding": f"vector({EMBEDDING_DIM})"
            }
        )

        # Crea indice HNSW per ricerca veloce
        self.pg_conn.create_index(
            index_name=f"idx_{TABLE_NAME}_embedding",
            table_name=TABLE_NAME,
            columns="embedding vector_cosine_ops",
            method="HNSW"
        )

        print(f"Tabella {TABLE_NAME} creata con successo")

    def populate(self, force: bool = False) -> None:
        """
        Popola la tabella con le descrizioni e gli embeddings.

        Args:
            force: Se True, ricrea la tabella anche se esiste
        """
        if force:
            print(f"Force mode: eliminando tabella {TABLE_NAME}...")
            self.pg_conn.drop_table(TABLE_NAME, cascade=True)

        # Crea tabella se non esiste
        if not self.pg_conn.table_exists(TABLE_NAME):
            self.create_table()

        # Estrai descrizioni dallo schema
        tables = self.schema_data.get("tables", [])

        if not tables:
            print("Nessuna tabella trovata nello schema")
            return

        print(f"Trovate {len(tables)} tabelle nello schema")

        for table in tables:
            table_name = table.get("table_name")
            description = table.get("description", "")

            if not table_name or not description:
                print(f"  Skipping tabella senza nome o descrizione: {table_name}")
                continue

            # Verifica se già esiste
            existing = self.pg_conn.execute_query(
                f"SELECT 1 FROM {TABLE_NAME} WHERE table_name = %s",
                (table_name,)
            )

            if existing:
                print(f"  {table_name}: già presente, skip")
                continue

            # Pulisci descrizione con intent extractor (se abilitato)
            if self.use_intent_extractor:
                print(f"  {table_name}: pulendo descrizione con intent extractor...")
                extracted_terms = self.intent_extractor.extract_categorical_intent(description)
                cleaned_description = " ".join(extracted_terms) if extracted_terms else description
                print(f"    - Originale: {len(description)} caratteri")
                print(f"    - Pulita: {len(cleaned_description)} caratteri ({len(extracted_terms)} termini)")
            else:
                cleaned_description = description

            # Genera embedding sulla descrizione pulita
            print(f"  {table_name}: generando embedding...")
            embedding = self.embedding_provider.embed(cleaned_description)

            # Inserisci descrizione pulita (non originale!)
            self.pg_conn.insert_data(
                TABLE_NAME,
                {
                    "table_name": table_name,
                    "description": cleaned_description,  # Descrizione pulita (per fuzzy matching)
                    "embedding": embedding.tolist()  # Embedding della descrizione pulita
                }
            )
            print(f"  {table_name}: inserito")

        print("Popolamento completato")

    def verify(self) -> None:
        """Verifica il contenuto della tabella."""
        if not self.pg_conn.table_exists(TABLE_NAME):
            print(f"Tabella {TABLE_NAME} non esiste")
            return

        results = self.pg_conn.execute_query(
            f"SELECT table_name, LENGTH(description) as desc_len FROM {TABLE_NAME}"
        )

        print(f"\nContenuto tabella {TABLE_NAME}:")
        for table_name, desc_len in results:
            print(f"  - {table_name}: {desc_len} caratteri")


if __name__ == "__main__":
    manager = TableDescriptionsManager()
    manager.populate(force=True)
    manager.verify()
