"""
YAML Configuration Loader
=========================

Module for loading SophIA configuration from YAML file.

Example:
    >>> from obelix.infrastructure.k8s import YamlConfig
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
    Simple YAML loader with dot-notation access.

    Example:
        >>> config = YamlConfig("config/agents.yaml")
        >>> chart_cfg = config.get("agents.chart_creator")
        >>> model_id = chart_cfg["model_id"]
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the loader by loading the YAML file.

        Args:
            yaml_path: Path to YAML file (absolute or relative to project root)

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file does not contain a valid dict
        """
        path = Path(yaml_path)
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            path = project_root / path

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            self._data = yaml.safe_load(f)

        if not isinstance(self._data, dict):
            raise ValueError(f"Invalid config: expected dict, got {type(self._data)}")

    def get(self, path: str) -> Any:
        """
        Access via dot-notation.

        Args:
            path: Path separated by dots (e.g. "agents.chart_creator")

        Returns:
            Raw value (dict/list/primitive)

        Raises:
            KeyError: If path not found
        """
        keys = path.split(".")
        current = self._data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(f"Path '{path}' not found")
            current = current[key]
        return current
