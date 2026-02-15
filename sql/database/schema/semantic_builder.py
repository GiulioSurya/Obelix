"""
Semantic Schema Builder
=======================

Classe utility per costruire schemi database arricchiti con semantic search.

Funzionalità:
1. Esegue semantic search per colonna su tabelle PostgreSQL con embeddings
2. Costruisce dizionario schema JSON con sample_values semanticamente rilevanti
3. Genera DDL SQL con commenti contenenti valori semantici

Workflow:
    Oracle Table (schema JSON) + PostgreSQL (vector DB) → Enriched Schema

Example:
    >>> from obelix.database.schema.semantic_builder import SemanticSchemaBuilder
    >>> builder = SemanticSchemaBuilder()
    >>> sql_schema = builder.enrich_schema(
    ...     semantic_query="finanziamenti per categorie deboli",
    ...     config={...}
    ... )
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel

from sql.connections.db_connection.postgres_connection import PostgresConnection
from obelix.ports.outbound.embedding_provider import AbstractEmbeddingProvider
from obelix.adapters.outbound.embedding.oci_embedding import OCIEmbeddingProvider
from obelix.infrastructure.k8s import YamlConfig
from sql.text.normalizer import TextNormalizer

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


load_dotenv()
infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))

# Configurazione colonne da indicizzare per tabella (importata da embed_tables.py)
DEFAULT_COLUMNS_TO_INDEX: Dict[str, List[str]] = infra_config.get(
    "database.schema.default_columns_to_index"
)


@dataclass
class TableMapping:
    """
    Mappa una tabella Oracle a una tabella PostgreSQL per semantic search.

    Attributes:
        oracle_table_name: Nome tabella Oracle (es. "VISTA_BILANCIO_ENTRATA_AI")
        postgres_table_name: Nome tabella Postgres con embeddings (es. "entrata_column_values")
        columns_to_search: Lista colonne da indicizzare semanticamente (opzionale)
    """
    oracle_table_name: str
    postgres_table_name: str
    columns_to_search: Optional[List[str]] = None


class SemanticSearchConfig(BaseModel):
    """
    Configurazione per semantic search con hybrid fuzzy reranking.
    Tutti i campi sono obbligatori - se manca una chiave nel YAML, ValidationError.
    """
    dynamic_retrieval: bool
    cardinality_threshold: int
    high_card_retrieval_k: int
    high_card_final_k: int
    low_card_retrieval_k: int
    low_card_final_k: int
    enable_fuzzy_reranking: bool
    semantic_weight: float
    fuzzy_weight: float
    rrf_k: int


# Default table mappings per bilancio schema
DEFAULT_TABLE_MAPPINGS = [
    TableMapping(
        oracle_table_name="VISTA_BILANCIO_ENTRATA_AI",
        postgres_table_name="entrata_column_values",
        columns_to_search=DEFAULT_COLUMNS_TO_INDEX['VISTA_BILANCIO_ENTRATA_AI']
    ),
    TableMapping(
        oracle_table_name="VISTA_BILANCIO_SPESA_AI",
        postgres_table_name="spesa_column_values",
        columns_to_search=DEFAULT_COLUMNS_TO_INDEX['VISTA_BILANCIO_SPESA_AI']
    )
]


class SemanticSchemaBuilder:
    """
    Costruisce schemi database arricchiti con semantic search su PostgreSQL.

    Questa classe combina:
    - Schema Oracle (da JSON file)
    - Embeddings PostgreSQL (vector DB)
    - Semantic search (via embedding provider)

    Per produrre schemi arricchiti con valori semanticamente rilevanti.

    Example:
        >>> builder = SemanticSchemaBuilder()
        >>> sql_schema = builder.enrich_schema(
        ...     semantic_query="finanziamenti per categorie deboli",
        ...     config={...}
        ... )
    """

    def __init__(
        self,
        postgres_conn: PostgresConnection,
        table_mappings: Optional[List[TableMapping]] = None,
        schema_json_path: Optional[Union[str, Path]] = None,
        embedding_provider: Optional[AbstractEmbeddingProvider] = None,
        text_normalizer: Optional[TextNormalizer] = None
    ):
        """
        Inizializza SemanticSchemaBuilder.

        Args:
            postgres_conn: Connessione PostgreSQL già inizializzata (OBBLIGATORIA).
            table_mappings: Lista mapping Oracle↔Postgres (default: DEFAULT_TABLE_MAPPINGS)
            schema_json_path: Path a schema JSON base (default: bilancio_schema.json)
            embedding_provider: Provider embedding (default: OCIEmbeddingProvider)
            text_normalizer: Normalizzatore testo per fuzzy search (default: TextNormalizer)
        """
        self.pg_conn = postgres_conn
        self.table_mappings = table_mappings or DEFAULT_TABLE_MAPPINGS
        self.schema_json_path = self._resolve_schema_path(schema_json_path)
        self.embedding_provider = embedding_provider or self._default_embedding_provider()
        self.normalizer = text_normalizer or TextNormalizer(keep_spaces=True)

        # Carica schema originale
        self.original_schema = self._load_schema()

    def _resolve_schema_path(self, path: Optional[Union[str, Path]]) -> Path:
        """Risolve il path al file schema JSON."""
        if path is None:
            # Default: bilancio_schema.json in src/database/cache/schema_cache
            return Path(__file__).parent.parent / "cache" / "schema_cache" / "bilancio_schema.json"
        return Path(path)

    def _default_embedding_provider(self) -> AbstractEmbeddingProvider:
        """Crea embedding provider di default (OCI Cohere)."""
        return OCIEmbeddingProvider(input_type="search_query")

    def _load_schema(self) -> Dict:
        """Carica lo schema JSON da file."""
        if not self.schema_json_path.exists():
            raise FileNotFoundError(f"Schema JSON non trovato: {self.schema_json_path}")

        with open(self.schema_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _fuzzy_search_column(
        self,
        fuzzy_query: str,
        column_name: str,
        postgres_table_name: str,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """
        Esegue fuzzy search su una colonna specifica usando rapidfuzz.

        Args:
            fuzzy_query: Query utente normalizzata (non migliorata) per fuzzy matching
            column_name: Nome della colonna
            postgres_table_name: Nome tabella Postgres
            top_k: Numero di risultati da restituire

        Returns:
            Lista di tuple (value, fuzzy_score) ordinate per score decrescente
        """
        if not RAPIDFUZZ_AVAILABLE:
            return []

        # Recupera tutti i valori della colonna
        values_query = f"""
            SELECT value
            FROM {postgres_table_name}
            WHERE column_name = %s
            ORDER BY value
        """
        results = self.pg_conn.execute_query(values_query, (column_name,))

        if not results:
            return []

        values = [row[0] for row in results]

        # Normalizza query per fuzzy matching
        normalized_query = self.normalizer.normalize(fuzzy_query)

        # Calcola fuzzy scores per ogni valore
        scored_values = []

        for value in values:
            # Normalizza valore per matching
            normalized_value = self.normalizer.normalize(value)

            # Combina 3 metriche fuzzy (come nel demo)
            partial = fuzz.partial_ratio(normalized_query, normalized_value, processor=None)
            token_set = fuzz.token_set_ratio(normalized_query, normalized_value, processor=None)
            wratio = fuzz.WRatio(normalized_query, normalized_value, processor=None)

            # Prendi il massimo tra le 3 metriche
            best_score = max(partial, token_set, wratio) / 100.0  # Normalizza 0-1
            scored_values.append((value, best_score))  # Salva valore ORIGINALE

        # Ordina per score decrescente, tiebreaker alfabetico per stabilità
        scored_values.sort(key=lambda x: (-x[1], x[0]))

        return scored_values[:top_k]

    def _weighted_rrf_merge(
        self,
        semantic_results: List[Tuple[str, float]],
        fuzzy_results: List[Tuple[str, float]],
        semantic_weight: float,
        fuzzy_weight: float,
        rrf_k: int,
        final_k: int
    ) -> List[str]:
        """
        Merge di semantic e fuzzy results usando Weighted Reciprocal Rank Fusion.

        Formula Weighted RRF:
            score(doc) = Σ [ weight_i / (rank_i + k) ] per ogni retriever i

        Args:
            semantic_results: Lista (value, similarity_score) da semantic search
            fuzzy_results: Lista (value, fuzzy_score) da fuzzy search
            semantic_weight: Peso per semantic (es. 0.75)
            fuzzy_weight: Peso per fuzzy (es. 0.25)
            rrf_k: Costante RRF (tipicamente 60)
            final_k: Numero di risultati finali

        Returns:
            Lista di valori ordinati per RRF score (top-k)
        """
        rrf_scores = {}

        # Semantic results
        for rank, (value, _) in enumerate(semantic_results, start=1):
            rrf_scores[value] = rrf_scores.get(value, 0.0) + semantic_weight / (rank + rrf_k)

        # Fuzzy results
        for rank, (value, _) in enumerate(fuzzy_results, start=1):
            rrf_scores[value] = rrf_scores.get(value, 0.0) + fuzzy_weight / (rank + rrf_k)

        # Ordina per RRF score decrescente, tiebreaker alfabetico per stabilità
        sorted_values = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))

        # Return solo i valori (no score)
        return [value for value, _ in sorted_values[:final_k]]

    def _map_oracle_type_to_sql(self, oracle_type: str) -> str:
        """
        Mappa tipi Oracle a tipi SQL standard.

        Args:
            oracle_type: Tipo Oracle (es. "NUMBER(10,2)", "VARCHAR2(100)")

        Returns:
            Tipo SQL standard equivalente
        """
        oracle_type_upper = oracle_type.upper()

        # NUMBER mappings
        if oracle_type_upper.startswith("NUMBER"):
            if "(" in oracle_type_upper:
                # NUMBER(n) o NUMBER(n,m)
                params = oracle_type_upper.split("(")[1].split(")")[0]
                if "," in params:
                    precision, scale = params.split(",")
                    return f"DECIMAL({precision},{scale})"
                else:
                    digits = int(params.strip())
                    if digits <= 4:
                        return "INT"
                    elif digits <= 18:
                        return "BIGINT"
                    else:
                        return "NUMERIC"
            return "NUMERIC"

        # VARCHAR2 -> VARCHAR
        if oracle_type_upper.startswith("VARCHAR2"):
            return oracle_type_upper.replace("VARCHAR2", "VARCHAR")

        # Default: mantieni il tipo originale
        return oracle_type

    def _is_numeric_column(self, column_type: str) -> bool:
        """
        Verifica se il tipo di colonna è numerico.

        Args:
            column_type: Tipo SQL della colonna

        Returns:
            True se la colonna è numerica, False altrimenti
        """
        numeric_types = {"INT", "BIGINT", "NUMERIC", "DECIMAL", "NUMBER"}
        base_type = column_type.upper().split("(")[0]
        return base_type in numeric_types

    def _determine_retrieval_params(
        self,
        column_name: str,
        postgres_table_name: str,
        config: SemanticSearchConfig
    ) -> Tuple[int, int]:
        """
        Determina retrieval_k e final_k in base alla cardinalità della colonna.

        Args:
            column_name: Nome della colonna
            postgres_table_name: Nome tabella Postgres
            config: Configurazione semantic search

        Returns:
            Tuple (retrieval_k, final_k)
        """
        retrieval_k = config.low_card_retrieval_k
        final_k = config.low_card_final_k

        if config.dynamic_retrieval:
            # Conta quanti valori ha la colonna (cardinalità)
            count_query = f"""
                SELECT COUNT(*)
                FROM {postgres_table_name}
                WHERE column_name = %s
            """
            count_result = self.pg_conn.execute_query(count_query, (column_name,))
            cardinality = count_result[0][0] if count_result else 0

            # Adatta retrieval_k e final_k in base alla cardinalità
            if cardinality >= config.cardinality_threshold:
                retrieval_k = config.high_card_retrieval_k
                final_k = config.high_card_final_k

        return retrieval_k, final_k

    def search_by_column(
        self,
        semantic_query: str,
        postgres_table_name: str,
        fuzzy_query: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Esegue hybrid search (semantic + fuzzy) filtrata per ogni column_name.

        Struttura tabella PostgreSQL:
        - column_name VARCHAR(128): nome colonna Oracle
        - value TEXT: valore distinto
        - embedding vector(1024): embedding OCI Cohere

        Args:
            semantic_query: Query migliorata per semantic search (embeddings)
            postgres_table_name: Nome tabella Postgres (es. "entrata_column_values")
            fuzzy_query: Query utente normalizzata per fuzzy search (opzionale, default: semantic_query)
            config: Dict configurazione semantic search (validato con Pydantic)
            columns: Lista colonne da cercare (opzionale, default: tutte le colonne indicizzate)

        Returns:
            Dict con {column_name: [list_of_semantic_values]}

        Raises:
            ValidationError: Se config manca di chiavi obbligatorie
        """
        if config is None:
            raise ValueError("config è obbligatorio - deve contenere semantic_search da agents.yaml")

        # Valida config con Pydantic
        validated_config = SemanticSearchConfig(**config)

        # Se fuzzy_query non è fornita, usa semantic_query (backward compatibility)
        if fuzzy_query is None:
            fuzzy_query = semantic_query

        # 1. Genera embedding della query semantica
        query_embedding = self.embedding_provider.embed(semantic_query)

        # 2. Determina colonne da cercare
        if columns is not None:
            # Usa solo colonne specificate (intersezione con quelle indicizzate in PostgreSQL)
            indexed_query = f"""
                SELECT DISTINCT column_name
                FROM {postgres_table_name}
                WHERE column_name = ANY(%s)
                ORDER BY column_name
            """
            indexed_columns = self.pg_conn.execute_query(indexed_query, (columns,))
            column_names = [col[0] for col in indexed_columns]
        else:
            # Default: tutte le colonne indicizzate
            columns_query = f"""
                SELECT DISTINCT column_name
                FROM {postgres_table_name}
                ORDER BY column_name
            """
            all_columns = self.pg_conn.execute_query(columns_query)
            column_names = [col[0] for col in all_columns]

        # 3. Dizionario per memorizzare i risultati
        semantic_results = {}

        # 4. Per ogni colonna, esegui hybrid search (semantic + fuzzy + RRF)
        for column_name in column_names:
            # Determina retrieval_k e final_k in base alla cardinalità
            retrieval_k, final_k = self._determine_retrieval_params(
                column_name, postgres_table_name, validated_config
            )

            # FASE 1: Semantic Search (retrieve candidati)
            similarity_query = f"""
                SELECT
                    value,
                    1 - (embedding <=> %s::vector) as similarity
                FROM {postgres_table_name}
                WHERE column_name = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            semantic_candidates = self.pg_conn.execute_query(
                similarity_query,
                (query_embedding.tolist(), column_name, query_embedding.tolist(), retrieval_k)
            )

            if not semantic_candidates:
                continue

            # FASE 2: Fuzzy Search (se abilitato)
            final_values = []

            if validated_config.enable_fuzzy_reranking and RAPIDFUZZ_AVAILABLE:
                fuzzy_candidates = self._fuzzy_search_column(
                    fuzzy_query=fuzzy_query,
                    column_name=column_name,
                    postgres_table_name=postgres_table_name,
                    top_k=retrieval_k
                )

                if fuzzy_candidates:
                    # FASE 3: Weighted RRF Merge
                    final_values = self._weighted_rrf_merge(
                        semantic_results=semantic_candidates,
                        fuzzy_results=fuzzy_candidates,
                        semantic_weight=validated_config.semantic_weight,
                        fuzzy_weight=validated_config.fuzzy_weight,
                        rrf_k=validated_config.rrf_k,
                        final_k=final_k
                    )
                else:
                    # Fallback: usa solo semantic
                    final_values = [value for value, _ in semantic_candidates[:final_k]]
            else:
                # Fuzzy disabilitato: usa solo semantic
                final_values = [value for value, _ in semantic_candidates[:final_k]]

            if final_values:
                semantic_results[column_name] = final_values

        return semantic_results

    def build_sql_schema(
        self,
        semantic_results_by_table: Dict[str, Dict[str, List[str]]],
        filter: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """
        Costruisce uno schema SQL string con sample_values come commenti.

        Args:
            semantic_results_by_table: Dict con struttura:
                {oracle_table_name: {column_name: [list_of_semantic_values]}}
            filter: Dict per filtrare tabelle e colonne (opzionale).
                    Chiavi = tabelle da includere, Valori = colonne da includere.

        Returns:
            String SQL con CREATE TABLE statements

        Example:
            >>> sql = builder.build_sql_schema(semantic_results)
            >>> print(sql)
            - Database Schema for Text-to-SQL
            CREATE TABLE VISTA_BILANCIO_ENTRATA_AI (
                DESCRIZIONE_CAP VARCHAR(140), - Descrizione capitolo
                - Sample values: "val1", "val2", "val3"
                ...
            );
        """
        # Output SQL (senza header superfluo)
        sql_output = []

        # Determina tabelle da processare
        tables_to_process = list(filter.keys()) if filter else None

        # Processa tabelle nell'ordine: prima SPESA, poi ENTRATA (come nel file originale)
        ordered_tables = []
        for oracle_table_name in ["VISTA_BILANCIO_SPESA_AI", "VISTA_BILANCIO_ENTRATA_AI"]:
            # Skip se filter specificato e tabella non presente
            if tables_to_process is not None and oracle_table_name not in tables_to_process:
                continue
            if oracle_table_name in semantic_results_by_table:
                ordered_tables.append((oracle_table_name, semantic_results_by_table[oracle_table_name]))

        for oracle_table_name, semantic_results in ordered_tables:
            # Trova la tabella nello schema originale
            target_table = None
            for table in self.original_schema.get("tables", []):
                if table["table_name"] == oracle_table_name:
                    target_table = table
                    break

            if not target_table:
                continue

            # Ottieni le colonne da indicizzare per questa tabella
            columns_to_index = DEFAULT_COLUMNS_TO_INDEX.get(oracle_table_name, [])

            # Determina colonne da includere (dal filter o tutte)
            columns_filter_set = set(filter[oracle_table_name]) if filter and oracle_table_name in filter else None

            # Inizio CREATE TABLE con descrizione tabella
            sql_output.append(f"CREATE TABLE {oracle_table_name} (")

            # Aggiungi descrizione tabella se presente
            table_description = target_table.get("description", "").strip()
            if table_description and table_description != "N/A":
                sql_output.append(f"    - {table_description}")

            columns = target_table.get("columns", [])
            for idx, column in enumerate(columns):
                column_name = column["column_name"]

                # Skip colonna se filter specificato e colonna non presente
                if columns_filter_set is not None and column_name not in columns_filter_set:
                    continue

                column_type = self._map_oracle_type_to_sql(column["type"])
                description = column.get("description", "N/A")

                # Costruisci riga colonna
                column_line = f"    {column_name} {column_type}"

                # Aggiungi commento descrizione
                if description and description != "N/A":
                    column_line += f", - {description}"
                else:
                    column_line += ","

                sql_output.append(column_line)

                # Determina se aggiungere sample values:
                # - Solo per colonne non numeriche
                # - O per colonne presenti in DEFAULT_COLUMNS_TO_INDEX
                should_include_samples = (
                    column_name in columns_to_index or
                    not self._is_numeric_column(column_type)
                )

                if should_include_samples:
                    # Aggiungi sample values come commento aggiuntivo
                    if column_name in semantic_results and semantic_results[column_name]:
                        # Usa sample values da semantic search (tutti i valori)
                        sample_values = semantic_results[column_name]
                        values_str = ", ".join([f'"{v}"' for v in sample_values])
                        sql_output.append(f"    - Sample values: {values_str}")
                    elif column.get("sample_values"):
                        # Usa sample values originali (tutti i valori)
                        sample_values = column["sample_values"]
                        values_str = ", ".join([f'"{v}"' for v in sample_values])
                        sql_output.append(f"    - Sample values: {values_str}")

            # Rimuovi ultima virgola e chiudi CREATE TABLE
            if sql_output[-1].endswith(","):
                sql_output[-1] = sql_output[-1][:-1]  # Rimuovi virgola finale

            sql_output.append(");")
            sql_output.append("")  # Riga vuota tra tabelle

        return "\n".join(sql_output)

    def enrich_schema(
        self,
        semantic_query: str,
        fuzzy_query: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        filter: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """
        Workflow completo: hybrid search (semantic + fuzzy) su tutte le tabelle mappate + build schema.

        Args:
            semantic_query: Query migliorata per semantic search (embeddings)
            fuzzy_query: Query utente normalizzata per fuzzy search (opzionale, default: semantic_query)
            config: Dict configurazione semantic search (validato con Pydantic)
            filter: Dict per filtrare tabelle e colonne (opzionale).
                    Chiavi = tabelle Oracle da processare.
                    Valori = colonne da includere per ogni tabella.
                    Es: {"VISTA_BILANCIO_SPESA_AI": ["ESERCIZIO", "CAPITOLO", ...]}

        Returns:
            String con DDL SQL arricchito semanticamente

        Raises:
            ValidationError: Se config manca di chiavi obbligatorie
        """
        if config is None:
            raise ValueError("config è obbligatorio - deve contenere semantic_search da agents.yaml")

        # Valida config con Pydantic (solleva ValidationError se mancano chiavi)
        SemanticSearchConfig(**config)

        # Se fuzzy_query non è fornita, usa semantic_query (backward compatibility)
        if fuzzy_query is None:
            fuzzy_query = semantic_query

        # Filtra table_mappings in base a filter.keys() (se fornito)
        if filter is not None:
            # Crea set per lookup O(1)
            table_names_set = set(filter.keys())
            # Filtra solo i mapping per le tabelle richieste
            mappings_to_process = [
                mapping for mapping in self.table_mappings
                if mapping.oracle_table_name in table_names_set
            ]
        else:
            # Default: usa tutte le tabelle mappate
            mappings_to_process = self.table_mappings

        # 1. Esegui hybrid search per ogni tabella mappata (filtrata se necessario)
        semantic_results_by_table = {}

        for mapping in mappings_to_process:
            # Ottieni colonne da cercare per questa tabella (dal filter o None per tutte)
            columns_to_search = filter.get(mapping.oracle_table_name) if filter else None

            results = self.search_by_column(
                semantic_query=semantic_query,
                fuzzy_query=fuzzy_query,
                postgres_table_name=mapping.postgres_table_name,
                config=config,
                columns=columns_to_search
            )

            semantic_results_by_table[mapping.oracle_table_name] = results

        # 2. Build schema SQL (passa filter per filtrare colonne nell'output)
        return self.build_sql_schema(semantic_results_by_table, filter=filter)
