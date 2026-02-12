"""
Modulo per gestione tabelle embedding PostgreSQL.
Crea e popola le tabelle per semantic search sui valori delle colonne.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.connections.db_connection.postgres_connection import PostgresConnection, get_postgres_connection
from src.embedding_providers.oci_embedding_provider import OCIEmbeddingProvider
from src.utils.text.normalizer import TextNormalizer


# Placeholder per valori speciali durante embedding
# Questi placeholder vengono usati SOLO per generare l'embedding,
# il valore originale (None, "", etc.) viene salvato nel DB
PLACEHOLDER_NULL = "[VALORE_NULL]"
PLACEHOLDER_EMPTY = "[VALORE_VUOTO]"


# Configurazione colonne da indicizzare per tabella
COLUMNS_TO_INDEX = {
    'VISTA_BILANCIO_ENTRATA_AI': [
        "DESCRIZIONE_CAP", "DESCRIZIONE_CAP_ABB", "DES_TITOLO", "DES_TIPOLOGIA",
        "DES_COD_LIVELLO_1", "DES_COD_LIVELLO_2", "DES_COD_LIVELLO_3",
        "DES_COD_LIVELLO_4", "DES_COD_LIVELLO_5", "DES_VINCOLO",
        "UNITA_ORGANIZZATIVA", "RESPONSABILE_UO", "DES_PROGRAMMA", "DES_PROGETTO",
        "RESPONSABILE", "SE_UNA_TANTUM", "DES_CENTRO_COSTO", "SE_RILEV_IVA",
        "SE_FUNZ_DELEG", "SE_CONTRIB_COMU", "SE_RISORSA_SIGNIF", "FLESSIBILITA",
        "DES_FLESSIBILITA", "DES_FATTORE", "DES_CENTRO", "DES_CGE",
        "DES_OPERA_LIGHT", "DES_FINANZIAMENTO_LIGHT", "OTTICA", "SETTORE"
    ],
    'VISTA_BILANCIO_SPESA_AI': [
        "DESCRIZIONE_CAP", "DESCRIZIONE_CAP_ABB", "DES_MISSIONE", "DES_PROGRAMMA_ARM",
        "DES_COD_LIVELLO_1", "DES_COD_LIVELLO_2", "DES_COD_LIVELLO_3",
        "DES_COD_LIVELLO_4", "DES_COD_LIVELLO_5", "DES_VINCOLO",
        "UNITA_ORGANIZZATIVA", "RESPONSABILE_UO", "DES_PROGRAMA", "DES_PROGETTO",
        "RESPONSABILE", "DES_CENTRO_COSTO", "FLESSIBILITA", "DES_FLESSIBILITA",
        "DES_FATTORE", "DES_CENTRO", "DES_CGU", "DES_OPERA_LIGHT",
        "DES_FINANZIAMENTO_LIGHT", "OTTICA", "SETTORE"
    ]
}


class EmbeddingTablesManager:
    """
    Gestisce la creazione e popolamento delle tabelle PostgreSQL
    per semantic search sui valori delle colonne.

    Struttura tabelle:
    - column_name: nome colonna (VARCHAR)
    - value: valore distinto (TEXT)
    - embedding: vettore 1024-dim (vector)

    Crea due tabelle:
    - entrata_column_values: per VISTA_BILANCIO_ENTRATA_AI
    - spesa_column_values: per VISTA_BILANCIO_SPESA_AI
    """

    def __init__(
        self,
        postgres_conn: Optional[PostgresConnection] = None,
        embedding_provider: Optional[OCIEmbeddingProvider] = None,
        schema_json_path: Optional[str] = None,
        text_normalizer: Optional[TextNormalizer] = None
    ):
        """
        Args:
            postgres_conn: Connessione PostgreSQL (default: usa get_postgres_connection())
            embedding_provider: Provider per generare embeddings (default: inizializzato lazy quando serve)
            schema_json_path: Path al file JSON con schema (default: bilancio_schema.json)
            text_normalizer: Normalizzatore testo per embeddings (default: TextNormalizer con keep_spaces=True)
        """
        self.pg_conn = postgres_conn or get_postgres_connection()

        # Embedding provider viene inizializzato lazy (solo quando serve per populate)
        self._embedding_provider = embedding_provider

        # Text normalizer per normalizzazione pre-embedding
        self.normalizer = text_normalizer or TextNormalizer(keep_spaces=True)

        # Path default al file schema
        if schema_json_path is None:
            # Path: src/database/cache/schema_cache/bilancio_schema.json
            schema_json_path = Path(__file__).parent.parent.parent / "database" / "cache" / "schema_cache" / "bilancio_schema.json"

        self.schema_json_path = Path(schema_json_path)
        self.schema_data = self._load_schema()

    @property
    def embedding_provider(self) -> OCIEmbeddingProvider:
        """
        Lazy initialization dell'embedding provider.
        Inizializza solo quando serve (durante populate).
        """
        if self._embedding_provider is None:
            print("🔧 Inizializzando OCI Embedding Provider...")
            self._embedding_provider = OCIEmbeddingProvider(
                input_type="search_document"  # Per indicizzare i valori
            )
        return self._embedding_provider

    def _load_schema(self) -> Dict:
        """Carica lo schema JSON da file."""
        if not self.schema_json_path.exists():
            raise FileNotFoundError(
                f"Schema JSON non trovato: {self.schema_json_path}"
            )

        with open(self.schema_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_tables(self, drop_if_exists: bool = False) -> None:
        """
        Crea le tabelle PostgreSQL per embeddings.

        Struttura:
        - column_name VARCHAR(128)
        - value TEXT
        - embedding vector(1024)

        Args:
            drop_if_exists: Se True, elimina le tabelle esistenti prima di crearle
        """
        print("🔧 Creazione tabelle embedding PostgreSQL...")

        # 1. Abilita estensione pgvector
        print("  ➤ Abilitando estensione pgvector...")
        self.pg_conn.enable_extension("vector")

        # 2. Crea tabella ENTRATA
        self._create_single_table("entrata_column_values", drop_if_exists)

        # 3. Crea tabella SPESA
        self._create_single_table("spesa_column_values", drop_if_exists)

        print("✅ Tabelle create con successo!")

    def _create_single_table(self, table_name: str, drop_if_exists: bool) -> None:
        """Crea una singola tabella con 3 colonne: column_name, value, embedding."""

        if drop_if_exists and self.pg_conn.table_exists(table_name):
            print(f"  ➤ Eliminando tabella esistente {table_name}...")
            self.pg_conn.drop_table(table_name, cascade=True)

        if not self.pg_conn.table_exists(table_name):
            print(f"  ➤ Creando tabella {table_name}...")

            # Crea tabella con SOLO 3 colonne
            self.pg_conn.create_table(table_name, {
                "column_name": "VARCHAR(128) NOT NULL",
                "value": "TEXT NOT NULL",
                "embedding": "vector(1024) NOT NULL"
            })

            # Aggiungi constraint UNIQUE su (column_name, value)
            self.pg_conn.execute_query(
                f"ALTER TABLE {table_name} ADD CONSTRAINT unique_{table_name}_col_val "
                f"UNIQUE (column_name, value)",
                fetch_results=False
            )

            # Crea indici
            print(f"  ➤ Creando indici per {table_name}...")

            # Indice HNSW per similarity search
            self.pg_conn.create_index(
                f"idx_{table_name}_hnsw",
                table_name,
                "embedding vector_cosine_ops",
                method="HNSW"
            )

            # Indice B-tree su column_name per filtri rapidi
            self.pg_conn.create_index(
                f"idx_{table_name}_column",
                table_name,
                "column_name"
            )
        else:
            print(f"  ℹ️  Tabella {table_name} già esistente, skip creazione")

    def extract_column_values(
        self,
        source_table: str
    ) -> Dict[str, List[str]]:
        """
        Estrae tutti i valori distinti per le colonne da indicizzare.

        Args:
            source_table: Nome tabella Oracle ('VISTA_BILANCIO_ENTRATA_AI' o 'VISTA_BILANCIO_SPESA_AI')

        Returns:
            Dict {column_name: [list_of_distinct_values]}
        """
        if source_table not in COLUMNS_TO_INDEX:
            raise ValueError(
                f"Tabella {source_table} non configurata. "
                f"Tabelle disponibili: {list(COLUMNS_TO_INDEX.keys())}"
            )

        columns_to_index = COLUMNS_TO_INDEX[source_table]

        # Trova la tabella nello schema JSON
        table_schema = None
        for table in self.schema_data.get("tables", []):
            if table["table_name"] == source_table:
                table_schema = table
                break

        if not table_schema:
            raise ValueError(f"Tabella {source_table} non trovata nello schema JSON")

        # Estrai valori per ogni colonna
        column_values = {}

        for column in table_schema.get("columns", []):
            column_name = column["column_name"]

            # Considera solo le colonne configurate
            if column_name not in columns_to_index:
                continue

            # Estrai sample_values (sono già i valori distinti)
            values = column.get("sample_values", [])

            if values:
                column_values[column_name] = values

        return column_values

    def populate_table(
        self,
        source_table: str,
        batch_size: int = 50,
        skip_existing: bool = True
    ) -> int:
        """
        Popola la tabella PostgreSQL con embeddings dei valori.

        Args:
            source_table: 'VISTA_BILANCIO_ENTRATA_AI' o 'VISTA_BILANCIO_SPESA_AI'
            batch_size: Numero di embeddings per batch (max 96 per API OCI)
            skip_existing: Se True, salta valori già presenti

        Returns:
            Numero di righe inserite
        """
        # Determina nome tabella Postgres
        if source_table == 'VISTA_BILANCIO_ENTRATA_AI':
            pg_table = "entrata_column_values"
        elif source_table == 'VISTA_BILANCIO_SPESA_AI':
            pg_table = "spesa_column_values"
        else:
            raise ValueError(f"Tabella non riconosciuta: {source_table}")

        print(f"\n📊 Popolamento tabella {pg_table} da {source_table}...")

        # Estrai valori da schema JSON
        column_values = self.extract_column_values(source_table)

        if not column_values:
            print(f"  ⚠️  Nessuna colonna da indicizzare trovata per {source_table}")
            return 0

        print(f"  ➤ Trovate {len(column_values)} colonne da indicizzare")

        total_inserted = 0

        # Per ogni colonna, genera embeddings e inserisci
        for column_name, values in column_values.items():
            print(f"\n  📝 Elaborando colonna: {column_name} ({len(values)} valori)")

            # Filtra valori già presenti se richiesto
            if skip_existing:
                existing_values_query = f"""
                    SELECT value FROM {pg_table}
                    WHERE column_name = %s
                """
                existing_rows = self.pg_conn.execute_query(
                    existing_values_query,
                    (column_name,)
                )
                existing_values = {row[0] for row in existing_rows}

                values_to_process = [v for v in values if v not in existing_values]

                if len(existing_values) > 0:
                    print(f"     ℹ️  {len(existing_values)} valori già presenti, skip")

                if not values_to_process:
                    print(f"     ✓ Tutti i valori già indicizzati")
                    continue

                values = values_to_process

            # Genera embeddings in batch
            print(f"     ⚙️  Generando {len(values)} embeddings...")

            # Suddividi in batch per rispettare limite API (96)
            for i in range(0, len(values), batch_size):
                batch_values = values[i:i + batch_size]

                try:
                    # NORMALIZZA i valori PRIMA di generare embeddings
                    normalized_values = self.normalizer.normalize_batch(batch_values)

                    # Sostituisci valori vuoti con placeholder per evitare errore API OCI
                    # "Some sentences in the inputs are empty"
                    values_for_embedding = []
                    for norm_val, orig_val in zip(normalized_values, batch_values):
                        if norm_val == "":
                            # Se normalizzato è vuoto, usa placeholder basato su originale
                            if orig_val is None:
                                values_for_embedding.append(PLACEHOLDER_NULL)
                            elif orig_val == "":
                                values_for_embedding.append(PLACEHOLDER_EMPTY)
                            else:
                                # Valore originale non vuoto ma normalizzato sì (es. solo punteggiatura)
                                values_for_embedding.append(PLACEHOLDER_EMPTY)
                        else:
                            values_for_embedding.append(norm_val)

                    # Genera embeddings per batch (su valori normalizzati con placeholder)
                    embeddings = self.embedding_provider.embed(values_for_embedding)

                    # Converti in lista se è singolo embedding
                    if isinstance(embeddings, np.ndarray):
                        embeddings = [embeddings]

                    # Prepara dati per insert (SOLO 3 colonne!)
                    # IMPORTANTE: salva valore ORIGINALE, ma embedding con placeholder se necessario
                    rows_to_insert = []
                    for original_value, embedding in zip(batch_values, embeddings):
                        rows_to_insert.append({
                            "column_name": column_name,
                            "value": original_value,  # ← Valore ORIGINALE (None, "", "I.R.A.P.", etc.)
                            "embedding": embedding.tolist()  # ← Embedding (con placeholder se vuoto)
                        })

                    # Insert batch in Postgres
                    self.pg_conn.insert_many(pg_table, rows_to_insert)

                    total_inserted += len(rows_to_insert)
                    print(f"     ✓ Batch {i//batch_size + 1}: {len(rows_to_insert)} righe inserite")

                except Exception as e:
                    print(f"     ❌ Errore batch {i//batch_size + 1}: {e}")
                    # Log valori problematici per debugging
                    print(f"        Valori nel batch: {batch_values[:5]}...")  # Prime 5 per non sovraccaricare
                    continue

        print(f"\n✅ Popolamento completato: {total_inserted} righe inserite in {pg_table}")
        return total_inserted

    def populate_all_tables(
        self,
        batch_size: int = 50,
        skip_existing: bool = True
    ) -> Tuple[int, int]:
        """
        Popola entrambe le tabelle (ENTRATA e SPESA).

        Args:
            batch_size: Numero di embeddings per batch
            skip_existing: Se True, salta valori già presenti

        Returns:
            Tuple (righe_entrata, righe_spesa)
        """
        print("Popolamento di tutte le tabelle embedding...\n")

        entrata_rows = self.populate_table(
            'VISTA_BILANCIO_ENTRATA_AI',
            batch_size=batch_size,
            skip_existing=skip_existing
        )

        spesa_rows = self.populate_table(
            'VISTA_BILANCIO_SPESA_AI',
            batch_size=batch_size,
            skip_existing=skip_existing
        )

        print(f"\nPopolamento completato!")
        print(f"   ENTRATA: {entrata_rows} righe")
        print(f"   SPESA: {spesa_rows} righe")
        print(f"   TOTALE: {entrata_rows + spesa_rows} righe")

        return entrata_rows, spesa_rows

    def get_table_stats(self) -> Dict:
        """
        Restituisce statistiche sulle tabelle embedding.

        Returns:
            Dict con statistiche per tabella
        """
        stats = {}

        for table_name in ["entrata_column_values", "spesa_column_values"]:
            if not self.pg_conn.table_exists(table_name):
                stats[table_name] = {"exists": False}
                continue

            # Conta righe totali
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            total_rows = self.pg_conn.execute_query(count_query)[0][0]

            # Conta colonne distinte
            columns_query = f"SELECT COUNT(DISTINCT column_name) FROM {table_name}"
            total_columns = self.pg_conn.execute_query(columns_query)[0][0]

            # Statistiche per colonna
            col_stats_query = f"""
                SELECT
                    column_name,
                    COUNT(*) as num_values
                FROM {table_name}
                GROUP BY column_name
                ORDER BY num_values DESC
            """
            col_stats = self.pg_conn.execute_query_dict(col_stats_query)

            stats[table_name] = {
                "exists": True,
                "total_rows": total_rows,
                "total_columns": total_columns,
                "columns": col_stats
            }

        return stats


if __name__ == "__main__":
    """
    Script per creare e popolare le tabelle embedding.

    Usage:
        python -m src.tools.utils.embed_tables
    """
    print("=" * 70)
    print("  EMBEDDING TABLES SETUP")
    print("=" * 70)

    # Inizializza manager
    manager = EmbeddingTablesManager()

    # Crea tabelle
    manager.create_tables(drop_if_exists=False)

    # Popola tabelle
    manager.populate_all_tables(batch_size=50, skip_existing=True)

    # Mostra statistiche
    print("\n" + "=" * 70)
    print("  STATISTICHE FINALI")
    print("=" * 70)
    stats = manager.get_table_stats()

    for table_name, table_stats in stats.items():
        if not table_stats.get("exists"):
            print(f"\n❌ {table_name}: NON ESISTENTE")
            continue

        print(f"\n✅ {table_name}:")
        print(f"   • Righe totali: {table_stats['total_rows']}")
        print(f"   • Colonne indicizzate: {table_stats['total_columns']}")

        if table_stats['columns']:
            print(f"\n   Top 5 colonne per numero valori:")
            for col in table_stats['columns'][:5]:
                print(f"      - {col['column_name']}: {col['num_values']} valori")