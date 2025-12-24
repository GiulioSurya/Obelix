"""
YAML Configuration Loader
=========================

Modulo per caricare configurazione SophIA da file YAML.

Example:
    >>> from src.k8s_config import YamlConfig
    >>> import os
    >>>
    >>> config = YamlConfig(os.getenv("CONFIG_PATH"))
    >>> chart_cfg = config.get("agents.chart_creator")
    >>> model_id = chart_cfg["model_id"]
"""

import yaml
from pathlib import Path
from typing import Any


class YamlConfig:
    """
    Semplice loader YAML con accesso via dot-notation.

    Example:
        >>> config = YamlConfig("config/agents.yaml")
        >>> chart_cfg = config.get("agents.chart_creator")
        >>> model_id = chart_cfg["model_id"]
    """

    def __init__(self, yaml_path: str):
        """
        Inizializza il loader caricando il file YAML.

        Args:
            yaml_path: Percorso al file YAML (assoluto o relativo alla root del progetto)

        Raises:
            FileNotFoundError: Se il file non esiste
            ValueError: Se il file non contiene un dict valido
        """
        path = Path(yaml_path)
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent
            path = project_root / path

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            self._data = yaml.safe_load(f)

        if not isinstance(self._data, dict):
            raise ValueError(f"Invalid config: expected dict, got {type(self._data)}")

    def get(self, path: str) -> Any:
        """
        Accesso via dot-notation.

        Args:
            path: Path separato da punti (es. "agents.chart_creator")

        Returns:
            Valore raw (dict/list/primitivo)

        Raises:
            KeyError: Se path non trovato
        """
        keys = path.split(".")
        current = self._data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(f"Path '{path}' not found")
            current = current[key]
        return current
