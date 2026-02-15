"""
Semantic Table Router
=====================

Modulo per routing intelligente delle tabelle database tramite semantic search + fuzzy matching.

Funzionalità:
- Semantic search: embedding query vs embedding descrizioni tabelle
- Fuzzy matching: rapidfuzz su descrizioni tabelle
- Media pesata: combina i due score normalizzati (0-1)

Workflow:
    User Query → [Semantic Search, Fuzzy Search] → Weighted Average → Selected Tables

Example:
    >>> from obelix.database.schema.router import SemanticTableRouter
    >>> router = SemanticTableRouter(postgres_conn)
    >>> tables = router.route("mostrami le entrate del 2024")
    >>> print(tables)  # ['VISTA_BILANCIO_ENTRATA_AI']
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sql.connections.db_connection.postgres_connection import PostgresConnection
from obelix.ports.outbound import AbstractEmbeddingProvider
from obelix.adapters.outbound.embedding.oci_embedding import OCIEmbeddingProvider
from sql.text.normalizer import TextNormalizer

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


# Nomi tabelle (per retrocompatibilità)
TABLE_ENTRATA = "VISTA_BILANCIO_ENTRATA_AI"
TABLE_SPESA = "VISTA_BILANCIO_SPESA_AI"

# Nome tabella Postgres per embeddings descrizioni
TABLE_DESCRIPTIONS_TABLE = "table_descriptions"


@dataclass
class RouterConfig:
    """
    Configurazione per SemanticTableRouter.

    Attributes:
        semantic_weight: Peso semantic search (default: 0.5)
        fuzzy_weight: Peso fuzzy search (default: 0.5)
        min_score_diff: Differenza minima tra top 2 score per selezionare solo top table.
                        Se diff < min_score_diff, ritorna entrambe le tabelle (default: 0.02)
    """
    semantic_weight: float = 0.5
    fuzzy_weight: float = 0.5
    min_score_diff: float = 0.02


class SemanticTableRouter:
    """
    Router che seleziona tabelle Oracle tramite media pesata (semantic + fuzzy).

    Combina:
    1. Semantic search: cosine similarity (0-1)
    2. Fuzzy search: rapidfuzz score (0-1)
    3. Media pesata: score finale normalizzato (0-1)

    Seleziona solo le tabelle con score >= threshold.

    Example:
        >>> router = SemanticTableRouter(postgres_conn)
        >>> tables = router.route("spese per assistenza sociale")
        >>> # ['VISTA_BILANCIO_SPESA_AI']
    """

    def __init__(
        self,
        postgres_conn: PostgresConnection,
        embedding_provider: Optional[AbstractEmbeddingProvider] = None,
        config: Optional[RouterConfig] = None,
        verbose: bool = False
    ):
        self.pg_conn = postgres_conn
        self.embedding_provider = embedding_provider or OCIEmbeddingProvider(input_type="search_query")
        self.config = config or RouterConfig()
        self.verbose = verbose
        self.normalizer = TextNormalizer(keep_spaces=True)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[Router] {message}")

    def _semantic_search(self, query: str) -> Dict[str, float]:
        """
        Semantic search: cosine similarity per ogni tabella.

        Returns:
            Dict {table_name: similarity} con score normalizzati (0-1)
        """
        query_embedding = self.embedding_provider.embed(query)

        sql = f"""
            SELECT
                table_name,
                1 - (embedding <=> %s::vector) as similarity
            FROM {TABLE_DESCRIPTIONS_TABLE}
        """

        results = self.pg_conn.execute_query(
            sql,
            (query_embedding.tolist(),)
        )

        return {row[0]: row[1] for row in results}

    def _fuzzy_search(self, query: str, intent_terms: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Fuzzy search parola-per-parola: per ogni termine query, calcola fuzzy con ogni termine descrizione.

        Algoritmo:
        1. Per ogni termine query → max fuzzy score tra tutti i termini descrizione
        2. Score finale tabella = media dei max score

        Args:
            query: Query utente (usata solo se intent_terms è None)
            intent_terms: Lista termini query già estratti (prioritario)

        Returns:
            Dict {table_name: fuzzy_score} con score normalizzati (0-1)
        """
        if not RAPIDFUZZ_AVAILABLE:
            self._log("rapidfuzz non disponibile, skip fuzzy search")
            return {}

        sql = f"SELECT table_name, description FROM {TABLE_DESCRIPTIONS_TABLE}"
        results = self.pg_conn.execute_query(sql)

        if not results:
            return {}

        # Se intent_terms non forniti, usa la query normalizzata (fallback legacy)
        if intent_terms is None:
            self._log("Intent terms non forniti, uso query normalizzata (fallback)")
            normalized_query = self.normalizer.normalize(query)
            query_terms = normalized_query.split()
        else:
            query_terms = intent_terms

        if not query_terms:
            self._log("Nessun termine query, skip fuzzy")
            return {}

        scores = {}

        for table_name, description in results:
            # Description è già pulita (termini separati da spazio)
            desc_terms = description.split()

            if not desc_terms:
                scores[table_name] = 0.0
                continue

            # Per ogni termine query, trova max score tra tutti termini descrizione
            term_max_scores = []
            for query_term in query_terms:
                max_score = 0.0
                for desc_term in desc_terms:
                    # Usa 4 funzioni fuzzy e prendi il max per robustezza
                    ratio_score = fuzz.ratio(query_term, desc_term, processor=None) / 100.0
                    wratio_score = fuzz.WRatio(query_term, desc_term, processor=None) / 100.0
                    token_set_score = fuzz.token_set_ratio(query_term, desc_term, processor=None) / 100.0
                    levenshtein_score = Levenshtein.normalized_similarity(query_term, desc_term, processor=None)

                    # Max tra le 4 funzioni
                    score = max(ratio_score, wratio_score, token_set_score, levenshtein_score)
                    max_score = max(max_score, score)
                term_max_scores.append(max_score)

            # Score finale = media dei max score
            scores[table_name] = sum(term_max_scores) / len(term_max_scores) if term_max_scores else 0.0

        return scores

    def _compute_final_scores(
        self,
        semantic_scores: Dict[str, float],
        fuzzy_scores: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Calcola score finale come media pesata.

        Formula:
            final = semantic_weight * semantic + fuzzy_weight * fuzzy

        Returns:
            Lista (table_name, final_score) ordinata per score decrescente
        """
        all_tables = set(semantic_scores.keys()) | set(fuzzy_scores.keys())
        final_scores = []

        for table in all_tables:
            semantic = semantic_scores.get(table, 0.0)
            fuzzy = fuzzy_scores.get(table, 0.0)

            final = (self.config.semantic_weight * semantic +
                     self.config.fuzzy_weight * fuzzy)

            final_scores.append((table, final))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores

    def route(self, query: str, intent_terms: Optional[List[str]] = None) -> List[str]:
        """
        Determina quali tabelle Oracle usare per la query.

        Seleziona la tabella con score massimo.
        Se la differenza tra i primi due score è < min_score_diff, ritorna entrambe.

        Args:
            query: Query utente in linguaggio naturale
            intent_terms: Termini estratti da QueryIntentExtractor (opzionale, per fuzzy matching)

        Returns:
            Lista nomi tabelle Oracle selezionate (1 o 2 tabelle)
        """
        self._log(f"Query: '{query}'")
        if intent_terms:
            self._log(f"Intent terms: {intent_terms}")

        # 1. Semantic search
        semantic_scores = self._semantic_search(query)
        self._log(f"Semantic: {semantic_scores}")

        # 2. Fuzzy search (con intent_terms se forniti)
        fuzzy_scores = self._fuzzy_search(query, intent_terms=intent_terms)
        self._log(f"Fuzzy: {fuzzy_scores}")

        # 3. Media pesata
        final_scores = self._compute_final_scores(semantic_scores, fuzzy_scores)
        self._log(f"Final scores: {final_scores}")

        # Fallback: se nessuna tabella, ritorna tutte
        if not final_scores:
            self._log("Nessuna tabella trovata, fallback a tutte")
            return [TABLE_ENTRATA, TABLE_SPESA]

        # 4. Controlla differenza tra primi due score
        if len(final_scores) >= 2:
            top_table, top_score = final_scores[0]
            second_table, second_score = final_scores[1]
            score_diff = top_score - second_score

            if score_diff < self.config.min_score_diff:
                self._log(f"Score diff {score_diff:.4f} < {self.config.min_score_diff}, ritorno entrambe")
                return [top_table, second_table]

        # 5. Ritorna solo la tabella con score massimo
        top_table = final_scores[0][0]
        self._log(f"Selezionata tabella con score massimo: {top_table}")
        return [top_table]


# =============================================================================
# Funzioni wrapper per retrocompatibilità
# =============================================================================

def route_tables(
    query: str,
    postgres_conn: Optional[PostgresConnection] = None,
    config: Optional[RouterConfig] = None,
    intent_terms: Optional[List[str]] = None,
    **kwargs,
) -> List[str]:
    """Wrapper retrocompatibile."""
    if postgres_conn is None:
        from obelix.connections.db_connection.postgres_connection import get_postgres_connection
        postgres_conn = get_postgres_connection()

    router = SemanticTableRouter(postgres_conn=postgres_conn, config=config, **kwargs)
    return router.route(query, intent_terms=intent_terms)


def route_tables_safe(
    query: str,
    postgres_conn: Optional[PostgresConnection] = None,
    config: Optional[RouterConfig] = None,
    intent_terms: Optional[List[str]] = None,
    fallback_all: bool = True,
) -> List[str]:
    """Versione safe con gestione eccezioni."""
    try:
        return route_tables(query, postgres_conn, config, intent_terms=intent_terms)
    except Exception:
        if fallback_all:
            return [TABLE_ENTRATA, TABLE_SPESA]
        return []
