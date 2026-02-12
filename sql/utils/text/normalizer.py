"""
Modulo per normalizzazione testo prima di embedding.
Utilizzato sia per normalizzare valori da indicizzare che query utente.
"""

import re
import unicodedata
from typing import Optional


class TextNormalizer:
    """
    Normalizza testo per embedding semantico.

    Operazioni applicate:
    1. Lowercase (tutto minuscolo)
    2. Rimozione accenti (à → a, è → e)
    3. Normalizzazione Unicode (NFKD)
    4. Rimozione caratteri speciali (punteggiatura, underscore, etc)
    5. Preservazione apostrofi (l'irap → l'irap)
    6. Normalizzazione spazi (multipli/leading/trailing)

    Esempi:
        >>> normalizer = TextNormalizer()
        >>> normalizer.normalize("I.R.A.P.")
        'irap'
        >>> normalizer.normalize("l'IRAP")
        "l'irap"
        >>> normalizer.normalize("Società à Responsabilità Limitata")
        'societa a responsabilita limitata'
        >>> normalizer.normalize("CODICE_FISCALE")
        'codicefiscale'
    """

    def __init__(self, keep_spaces: bool = True):
        """
        Args:
            keep_spaces: Se True mantiene spazi tra parole, altrimenti li rimuove
        """
        self.keep_spaces = keep_spaces

        # Pattern regex per rimuovere caratteri speciali
        # Mantiene: lettere (unicode), numeri, spazi, apostrofi
        # Nota: \w include underscore, quindi lo rimuoviamo esplicitamente
        if keep_spaces:
            # Rimuove tutto tranne lettere, numeri, spazi e apostrofi (')
            self.special_chars_pattern = re.compile(r"[^a-zA-Z0-9\s'%]", re.UNICODE)
            self.underscore_pattern = re.compile(r'_')
        else:
            # Rimuove tutto tranne lettere, numeri, apostrofi e %
            self.special_chars_pattern = re.compile(r"[^a-zA-Z0-9'%]", re.UNICODE)
            self.underscore_pattern = re.compile(r'_')

        # Pattern per normalizzare spazi multipli
        self.multiple_spaces_pattern = re.compile(r'\s+')

    def normalize(self, text: Optional[str]) -> str:
        """
        Normalizza una stringa per embedding.

        Args:
            text: Testo da normalizzare

        Returns:
            Testo normalizzato (può essere stringa vuota se input vuoto)

        Examples:
            >>> normalizer = TextNormalizer()
            >>> normalizer.normalize("I.R.A.P.")
            'irap'
            >>> normalizer.normalize("Società à Responsabilità Limitata")
            'societa a responsabilita limitata'
            >>> normalizer.normalize("  TEST___VALUE  ")
            'testvalue'
            >>> normalizer.normalize("")
            ''
            >>> normalizer.normalize(None)
            ''
        """
        # Se None o non stringa, restituisci stringa vuota
        if text is None or not isinstance(text, str):
            return ""

        # Se stringa vuota, restituisci stringa vuota (non filtriamo)
        if text == "":
            return ""

        # 1. Lowercase
        normalized = text.lower()

        # 2. Normalizzazione Unicode NFKD (decompone caratteri accentati)
        normalized = unicodedata.normalize('NFKD', normalized)

        # 3. Rimozione accenti (caratteri combining)
        normalized = ''.join([
            char for char in normalized
            if not unicodedata.combining(char)
        ])

        # 4. Rimozione underscore (prima dei caratteri speciali)
        normalized = self.underscore_pattern.sub('', normalized)

        # 5. Rimozione caratteri speciali (mantiene solo lettere, numeri, spazi)
        normalized = self.special_chars_pattern.sub('', normalized)

        # 5. Normalizzazione spazi
        if self.keep_spaces:
            # Sostituisci spazi multipli con singolo spazio
            normalized = self.multiple_spaces_pattern.sub(' ', normalized)
            # Rimuovi spazi leading/trailing
            normalized = normalized.strip()
        else:
            # Rimuovi tutti gli spazi
            normalized = normalized.replace(' ', '')

        return normalized

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """
        Normalizza un batch di stringhe.

        Args:
            texts: Lista di testi da normalizzare

        Returns:
            Lista di testi normalizzati

        Examples:
            >>> normalizer = TextNormalizer()
            >>> normalizer.normalize_batch(["I.R.A.P.", "I.V.A.", "CODICE_FISCALE"])
            ['irap', 'iva', 'codicefiscale']
        """
        return [self.normalize(text) for text in texts]


# Singleton instance per uso globale
_default_normalizer: Optional[TextNormalizer] = None


def get_normalizer(keep_spaces: bool = True) -> TextNormalizer:
    """
    Restituisce singleton instance di TextNormalizer.

    Args:
        keep_spaces: Se True mantiene spazi tra parole

    Returns:
        Instance di TextNormalizer
    """
    global _default_normalizer

    if _default_normalizer is None:
        _default_normalizer = TextNormalizer(keep_spaces=keep_spaces)

    return _default_normalizer


if __name__ == "__main__":
    """Test rapidi della normalizzazione."""

    normalizer = TextNormalizer()

    test_cases = [
        "I.R.A.P.",
        "Società à Responsabilità Limitata",
        "CODICE_FISCALE",
        "TEST___VALUE",
        "  spazi   multipli  ",
        "àèéìòù ÄËÏÖÜ",
        "Test-con-trattini",
        "Test.Con.Punti",
        "123 numeri 456",
    ]

    print("=" * 70)
    print("TEST NORMALIZZAZIONE TESTO")
    print("=" * 70)

    for original in test_cases:
        normalized = normalizer.normalize(original)
        print(f"\n'{original}'")
        print(f"  → '{normalized}'")

    print("\n" + "=" * 70)
    print("TEST BATCH")
    print("=" * 70)

    batch = ["I.R.A.P.", "I.V.A.", "CODICE_FISCALE"]
    normalized_batch = normalizer.normalize_batch(batch)
    print(f"\nOriginal: {batch}")
    print(f"Normalized: {normalized_batch}")
