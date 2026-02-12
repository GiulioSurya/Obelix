"""
Query Intent Extractor
=====================

Estrae keyword rilevanti da query conversazionali in italiano per matching categorico.

Utilizza spaCy con modello italiano small per:
1. Named Entity Recognition (NER): identifica e scarta date, numeri
2. POS tagging: mantiene solo sostantivi, nomi propri, aggettivi
3. Stopwords filtering: rimuove parole funzionali
4. Lemmatization: normalizza forme varianti

Pipeline:
    Query conversazionale � NER + POS tagging � Filtered tokens � Lemmatized intent terms

Example:
    >>> from src.semantic.analysis.query_intent_extractor import QueryIntentExtractor
    >>> extractor = QueryIntentExtractor()
    >>> intent_terms = extractor.extract_categorical_intent(
    ...     "quanto abbiamo speso di l'irap per il 2025?"
    ... )
    >>> print(intent_terms)  # ['irap']

Use Case:
    Migliora fuzzy matching filtrando rumore dalle query conversazionali:
    - Prima: fuzzy("quanto abbiamo speso di l'irap per il 2025?", "IRAP GETTITO ARRETRATO") � 0.42
    - Dopo: fuzzy("irap", "IRAP GETTITO ARRETRATO") � 0.95

Requirements:
    pip install spacy
    python -m spacy download it_core_news_sm
"""

from typing import List, Set, Optional, Tuple
from dataclasses import dataclass

try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
1

@dataclass
class IntentExtractionConfig:
    """
    Configurazione per estrazione intent da query conversazionali.

    Attributes:
        model_name: Nome modello spaCy (default: it_core_news_sm)
        relevant_pos_tags: POS tags da mantenere (default: NOUN, PROPN, ADJ)
        ignore_entity_types: Entity types da scartare (default: DATE, TIME, CARDINAL, ORDINAL)
        custom_stopwords: Set stopwords custom oltre a quelle spaCy (default: None)
        use_default_custom_stopwords: Se True, usa DEFAULT_CUSTOM_STOPWORDS (default: True)
        enable_lemmatization: Se True, usa lemma normalizzato (default: True)
        enable_noun_chunks: Se True, estrae anche noun chunks multi-word (default: True)
        min_token_length: Lunghezza minima token (default: 2)
    """
    model_name: str = "it_core_news_sm"
    relevant_pos_tags: Set[str] = None
    ignore_entity_types: Set[str] = None
    custom_stopwords: Optional[Set[str]] = None
    use_default_custom_stopwords: bool = True
    enable_lemmatization: bool = True
    enable_noun_chunks: bool = True
    min_token_length: int = 2

    def __post_init__(self):
        """Inizializza defaults per set mutabili."""
        if self.relevant_pos_tags is None:
            # NOUN: sostantivi comuni (es. "spesa", "programma")
            # PROPN: nomi propri (es. "IRAP", "Italia")
            # ADJ: aggettivi (es. "sociale", "arretrato")
            self.relevant_pos_tags = {"NOUN", "PROPN", "ADJ"}

        if self.ignore_entity_types is None:
            # DATE: date (es. "2025", "gennaio 2024")
            # TIME: orari (es. "10:30")
            # CARDINAL: numeri cardinali (es. "1000", "cinquanta")
            # ORDINAL: numeri ordinali (es. "primo", "terzo")
            self.ignore_entity_types = {"DATE", "TIME", "CARDINAL", "ORDINAL"}


class QueryIntentExtractor:
    """
    Estrae termini italiani rilevanti da query conversazionali per matching categorico.

    Utilizza spaCy con modello italiano small per NLP-based extraction:
    - NER per identificare e scartare entit� temporali/numeriche
    - POS tagging per mantenere solo sostantivi/nomi propri/aggettivi
    - Stopwords filtering per rimuovere parole funzionali
    - Lemmatization per normalizzare forme varianti

    Attributes:
        config: Configurazione estrazione intent
        nlp: Modello spaCy (lazy-loaded al primo utilizzo)

    Example:
        >>> extractor = QueryIntentExtractor()
        >>> terms = extractor.extract_categorical_intent("quanto abbiamo speso di l'irap?")
        >>> print(terms)  # ['irap']

        >>> # Con noun chunks multi-word
        >>> terms = extractor.extract_categorical_intent("programma assistenza sociale 2024")
        >>> print(terms)  # ['programma assistenza sociale']
    """

    # Stopwords custom aggiuntive specifiche per query conversazionali
    # (oltre alle stopwords standard di spaCy)
    DEFAULT_CUSTOM_STOPWORDS = {
        # Interrogativi
        'quanto', 'quanta', 'quanti', 'quante',

        # Verbi query comuni (backup se POS tagger fallisce)
        'speso', 'spesa', 'costo', 'costato', 'costata',
        'pagato', 'pagata', 'importo', 'ammontare', 'totale',

        # Altri
        'anno', 'mese'
    }

    def __init__(self, config: Optional[IntentExtractionConfig] = None):
        """
        Inizializza extractor con configurazione opzionale.

        Args:
            config: Configurazione custom (default: IntentExtractionConfig())

        Raises:
            ImportError: Se spaCy non � installato
            OSError: Se modello italiano non � scaricato
        """
        if not SPACY_AVAILABLE:
            raise ImportError(
                "spaCy non installato. Installare con:\n"
                "  pip install spacy\n"
                "  python -m spacy download it_core_news_sm"
            )

        self.config = config or IntentExtractionConfig()
        self._nlp: Optional[Language] = None

        # Inizializza custom stopwords in base al flag use_default_custom_stopwords
        if self.config.use_default_custom_stopwords:
            self._custom_stopwords = self.DEFAULT_CUSTOM_STOPWORDS.copy()
        else:
            self._custom_stopwords = set()

        # Aggiungi eventuali stopwords custom aggiuntive
        if self.config.custom_stopwords:
            self._custom_stopwords.update(self.config.custom_stopwords)

    @property
    def nlp(self) -> Language:
        """Lazy-load modello spaCy al primo utilizzo."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.config.model_name)
            except OSError as e:
                raise OSError(
                    f"Modello spaCy '{self.config.model_name}' non trovato.\n"
                    f"Installare con: python -m spacy download {self.config.model_name}"
                ) from e
        return self._nlp

    def extract_categorical_intent(
        self,
        query: str,
        additional_stopwords: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Estrae termini italiani rilevanti per matching categorico.

        Pipeline:
        1. spaCy NER: identifica entit� DATE/CARDINAL e marca per scarto
        2. POS tagging: mantieni solo NOUN/PROPN/ADJ
        3. Stopwords: filtra parole funzionali (spaCy + custom)
        4. Lemmatization: normalizza forme (es. "l'irap" � "irap")
        5. Noun chunks (opzionale): estrae anche multi-word (es. "programma sociale")

        Args:
            query: Query conversazionale in italiano
            additional_stopwords: Stopwords aggiuntive per questo specifico caso

        Returns:
            Lista termini italiani rilevanti (es. ['irap', 'gettito arretrato'])
            Ritorna lista vuota se query vuota o nessun termine rilevante

        Example:
            >>> extractor.extract_categorical_intent("quanto abbiamo speso di l'irap per il 2025?")
            ['irap']

            >>> extractor.extract_categorical_intent("costo programma assistenza sociale 2024")
            ['programma assistenza sociale', 'costo']  # se noun_chunks abilitato

            >>> extractor.extract_categorical_intent("")
            []
        """
        if not query or not query.strip():
            return []

        doc = self.nlp(query)

        # Merge stopwords
        stopwords_set = self._custom_stopwords.copy()
        if additional_stopwords:
            stopwords_set.update(additional_stopwords)

        # Step 1: Identifica span da ignorare (entit� DATE/CARDINAL/etc.)
        ignore_spans = {
            (ent.start, ent.end)
            for ent in doc.ents
            if ent.label_ in self.config.ignore_entity_types
        }

        # Step 2: Estrai noun chunks multi-word (se abilitato)
        noun_chunks = []
        if self.config.enable_noun_chunks:
            for chunk in doc.noun_chunks:
                # Salta se chunk contiene entit� ignorata
                if any(
                    chunk.start <= idx < chunk.end
                    for start, end in ignore_spans
                    for idx in range(start, end)
                ):
                    continue

                # Lemmatizza e normalizza chunk
                chunk_lemma = self._normalize_chunk(chunk, stopwords_set)
                if chunk_lemma and len(chunk_lemma) >= self.config.min_token_length:
                    noun_chunks.append(chunk_lemma)

        # Step 3: Estrai singoli token rilevanti
        single_tokens = []
        for i, token in enumerate(doc):
            # Salta se parte di entit� ignorata
            if any(start <= i < end for start, end in ignore_spans):
                continue

            # Salta se parte di noun chunk gi� estratto
            if self.config.enable_noun_chunks:
                if any(chunk.start <= i < chunk.end for chunk in doc.noun_chunks):
                    continue

            # Mantieni solo POS rilevanti
            if token.pos_ not in self.config.relevant_pos_tags:
                continue

            # Filtra stopwords
            if token.is_stop or token.text.lower() in stopwords_set:
                continue

            # Filtra punteggiatura/spazi
            if token.is_punct or token.is_space:
                continue

            # Filtra token troppo corti
            if len(token.text) < self.config.min_token_length:
                continue

            # Usa lemma o testo originale
            term = token.lemma_.lower() if self.config.enable_lemmatization else token.text.lower()
            single_tokens.append(term)

        # Step 4: Combina noun chunks (prioritari) e single tokens
        # Noun chunks vanno prima perch� pi� specifici (es. "programma sociale" prima di "programma")
        result = noun_chunks + single_tokens

        # Rimuovi duplicati mantenendo ordine
        seen = set()
        unique_result = []
        for term in result:
            if term not in seen:
                seen.add(term)
                unique_result.append(term)

        return unique_result

    def extract_with_metadata(
        self,
        query: str,
        additional_stopwords: Optional[Set[str]] = None
    ) -> List[Tuple[str, str, str]]:
        """
        Come extract_categorical_intent ma ritorna anche metadati POS/entity.

        Utile per debugging e analisi qualit� extraction.

        Args:
            query: Query conversazionale in italiano
            additional_stopwords: Stopwords aggiuntive opzionali

        Returns:
            Lista tuple (termine, pos_tag, tipo) dove tipo = "token" o "chunk"

        Example:
            >>> extractor.extract_with_metadata("quanto abbiamo speso di l'irap?")
            [('irap', 'PROPN', 'token')]

            >>> extractor.extract_with_metadata("programma assistenza sociale")
            [('programma assistenza sociale', 'NOUN_CHUNK', 'chunk'),
             ('programma', 'NOUN', 'token'), ...]
        """
        if not query or not query.strip():
            return []

        doc = self.nlp(query)

        stopwords_set = self._custom_stopwords.copy()
        if additional_stopwords:
            stopwords_set.update(additional_stopwords)

        ignore_spans = {
            (ent.start, ent.end)
            for ent in doc.ents
            if ent.label_ in self.config.ignore_entity_types
        }

        result = []

        # Noun chunks
        if self.config.enable_noun_chunks:
            for chunk in doc.noun_chunks:
                if any(
                    chunk.start <= idx < chunk.end
                    for start, end in ignore_spans
                    for idx in range(start, end)
                ):
                    continue

                chunk_lemma = self._normalize_chunk(chunk, stopwords_set)
                if chunk_lemma and len(chunk_lemma) >= self.config.min_token_length:
                    result.append((chunk_lemma, "NOUN_CHUNK", "chunk"))

        # Single tokens
        for i, token in enumerate(doc):
            if any(start <= i < end for start, end in ignore_spans):
                continue

            if self.config.enable_noun_chunks:
                if any(chunk.start <= i < chunk.end for chunk in doc.noun_chunks):
                    continue

            if token.pos_ not in self.config.relevant_pos_tags:
                continue

            if token.is_stop or token.text.lower() in stopwords_set:
                continue

            if token.is_punct or token.is_space:
                continue

            if len(token.text) < self.config.min_token_length:
                continue

            term = token.lemma_.lower() if self.config.enable_lemmatization else token.text.lower()
            result.append((term, token.pos_, "token"))

        return result

    def _normalize_chunk(self, chunk, stopwords_set: Set[str]) -> str:
        """
        Normalizza noun chunk rimuovendo stopwords e lemmatizzando.

        Args:
            chunk: spaCy Span (noun chunk)
            stopwords_set: Set stopwords da filtrare

        Returns:
            Chunk normalizzato (lemmatizzato e lowercase) o stringa vuota se tutto stopwords
        """
        # Filtra stopwords e lemmatizza ogni token
        tokens = [
            token.lemma_.lower() if self.config.enable_lemmatization else token.text.lower()
            for token in chunk
            if not token.is_stop
            and token.text.lower() not in stopwords_set
            and not token.is_punct
            and not token.is_space
        ]

        return " ".join(tokens) if tokens else ""


# Factory function per creazione rapida
def create_intent_extractor(
    model_name: str = "it_core_news_sm",
    enable_noun_chunks: bool = True,
    **kwargs
) -> QueryIntentExtractor:
    """
    Factory function per creare QueryIntentExtractor con config custom.

    Args:
        model_name: Nome modello spaCy (default: it_core_news_sm)
        enable_noun_chunks: Se True, estrae anche multi-word (default: True)
        **kwargs: Altri parametri IntentExtractionConfig

    Returns:
        QueryIntentExtractor configurato

    Example:
        >>> extractor = create_intent_extractor(enable_noun_chunks=False)
        >>> terms = extractor.extract_categorical_intent("query...")
    """
    config = IntentExtractionConfig(
        model_name=model_name,
        enable_noun_chunks=enable_noun_chunks,
        **kwargs
    )
    return QueryIntentExtractor(config=config)

if __name__ == "__main__":
    extractor = create_intent_extractor(enable_noun_chunks=False)
    terms = extractor.extract_categorical_intent("Qual è l’importo di cassa residua ancora disponibile, riferito al Settore COMUNALE, filtrato per i servizi associati, diviso per titoli di bilancio?”")
    print(terms)