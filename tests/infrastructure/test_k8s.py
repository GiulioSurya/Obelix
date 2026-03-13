"""Tests for obelix.infrastructure.k8s -- YamlConfig loader."""

import pytest

from obelix.infrastructure.k8s import YamlConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_yaml(tmp_path) -> str:
    """Create a simple YAML config file and return its path."""
    content = """\
database:
  host: localhost
  port: 5432
  credentials:
    user: admin
    password: secret
agents:
  chart_creator:
    model_id: gpt-4
    temperature: 0.7
  sql_generator:
    model_id: claude-3
    max_tokens: 4096
features:
  - name: auth
    enabled: true
  - name: cache
    enabled: false
top_level_value: 42
"""
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def nested_yaml(tmp_path) -> str:
    """Create a deeply nested YAML config file."""
    content = """\
level1:
  level2:
    level3:
      level4:
        value: deep
"""
    path = tmp_path / "nested.yaml"
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def empty_dict_yaml(tmp_path) -> str:
    """Create a YAML file containing an empty dict."""
    path = tmp_path / "empty_dict.yaml"
    path.write_text("{}", encoding="utf-8")
    return str(path)


@pytest.fixture
def list_yaml(tmp_path) -> str:
    """Create a YAML file whose root is a list (invalid for YamlConfig)."""
    path = tmp_path / "list.yaml"
    path.write_text("- item1\n- item2\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def scalar_yaml(tmp_path) -> str:
    """Create a YAML file whose root is a scalar (invalid for YamlConfig)."""
    path = tmp_path / "scalar.yaml"
    path.write_text("just a string\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def null_yaml(tmp_path) -> str:
    """Create a YAML file that parses to None."""
    path = tmp_path / "null.yaml"
    path.write_text("", encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# TestYamlConfigInit -- constructor / loading
# ---------------------------------------------------------------------------


class TestYamlConfigInit:
    """Tests for YamlConfig.__init__."""

    def test_load_valid_yaml(self, simple_yaml: str):
        config = YamlConfig(simple_yaml)
        assert config._data is not None
        assert isinstance(config._data, dict)

    def test_file_not_found_raises(self, tmp_path):
        nonexistent = str(tmp_path / "does_not_exist.yaml")
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            YamlConfig(nonexistent)

    def test_root_is_list_raises_value_error(self, list_yaml: str):
        with pytest.raises(ValueError, match="expected dict"):
            YamlConfig(list_yaml)

    def test_root_is_scalar_raises_value_error(self, scalar_yaml: str):
        with pytest.raises(ValueError, match="expected dict"):
            YamlConfig(scalar_yaml)

    def test_root_is_none_raises_value_error(self, null_yaml: str):
        """An empty YAML file parses to None, which is not a dict."""
        with pytest.raises(ValueError, match="expected dict"):
            YamlConfig(null_yaml)

    def test_empty_dict_loads_successfully(self, empty_dict_yaml: str):
        config = YamlConfig(empty_dict_yaml)
        assert config._data == {}

    def test_absolute_path_used_directly(self, simple_yaml: str):
        """An absolute path should be used as-is without prepending project root."""
        config = YamlConfig(simple_yaml)
        # If loading succeeded with the absolute tmp_path, the path was not mangled
        assert "database" in config._data


# ---------------------------------------------------------------------------
# TestYamlConfigGet -- dot-notation access
# ---------------------------------------------------------------------------


class TestYamlConfigGet:
    """Tests for YamlConfig.get() dot-notation path resolution."""

    def test_top_level_key(self, simple_yaml: str):
        config = YamlConfig(simple_yaml)
        assert config.get("top_level_value") == 42

    def test_nested_key(self, simple_yaml: str):
        config = YamlConfig(simple_yaml)
        result = config.get("database.host")
        assert result == "localhost"

    def test_nested_two_levels(self, simple_yaml: str):
        config = YamlConfig(simple_yaml)
        result = config.get("database.credentials.user")
        assert result == "admin"

    def test_returns_dict_for_intermediate_path(self, simple_yaml: str):
        config = YamlConfig(simple_yaml)
        result = config.get("database.credentials")
        assert isinstance(result, dict)
        assert result == {"user": "admin", "password": "secret"}

    def test_returns_list_value(self, simple_yaml: str):
        config = YamlConfig(simple_yaml)
        features = config.get("features")
        assert isinstance(features, list)
        assert len(features) == 2

    def test_returns_entire_section(self, simple_yaml: str):
        config = YamlConfig(simple_yaml)
        agents = config.get("agents")
        assert isinstance(agents, dict)
        assert "chart_creator" in agents
        assert "sql_generator" in agents

    def test_deeply_nested_access(self, nested_yaml: str):
        config = YamlConfig(nested_yaml)
        assert config.get("level1.level2.level3.level4.value") == "deep"

    def test_missing_key_raises_key_error(self, simple_yaml: str):
        config = YamlConfig(simple_yaml)
        with pytest.raises(KeyError, match="not found"):
            config.get("nonexistent")

    def test_missing_nested_key_raises_key_error(self, simple_yaml: str):
        config = YamlConfig(simple_yaml)
        with pytest.raises(KeyError, match="not found"):
            config.get("database.nonexistent_key")

    def test_path_through_non_dict_raises_key_error(self, simple_yaml: str):
        """Traversing through a scalar (e.g., 'database.host.something') should fail."""
        config = YamlConfig(simple_yaml)
        with pytest.raises(KeyError, match="not found"):
            config.get("database.host.deep")

    def test_path_through_list_raises_key_error(self, simple_yaml: str):
        """Traversing into a list value raises KeyError (no index support)."""
        config = YamlConfig(simple_yaml)
        with pytest.raises(KeyError, match="not found"):
            config.get("features.0")

    def test_empty_dict_any_key_raises(self, empty_dict_yaml: str):
        config = YamlConfig(empty_dict_yaml)
        with pytest.raises(KeyError, match="not found"):
            config.get("anything")

    def test_numeric_string_key(self, tmp_path):
        """Keys that look like numbers should work as strings."""
        path = tmp_path / "numeric_keys.yaml"
        # Note: in YAML, unquoted 'yes' is parsed as boolean True,
        # and unquoted 123 is parsed as int. Quote both to keep them as strings.
        path.write_text("'123':\n  value: 'yes'\n", encoding="utf-8")
        config = YamlConfig(str(path))
        assert config.get("123.value") == "yes"

    def test_get_returns_primitive_types(self, simple_yaml: str):
        config = YamlConfig(simple_yaml)
        # int
        assert config.get("database.port") == 5432
        assert isinstance(config.get("database.port"), int)
        # float
        assert config.get("agents.chart_creator.temperature") == 0.7
        # str
        assert config.get("database.host") == "localhost"

    def test_agent_config_access_pattern(self, simple_yaml: str):
        """Test the documented usage pattern from the module docstring."""
        config = YamlConfig(simple_yaml)
        chart_cfg = config.get("agents.chart_creator")
        assert chart_cfg["model_id"] == "gpt-4"
        assert chart_cfg["temperature"] == 0.7


# ---------------------------------------------------------------------------
# TestYamlConfigEdgeCases
# ---------------------------------------------------------------------------


class TestYamlConfigEdgeCases:
    """Edge cases and special YAML content."""

    def test_unicode_content(self, tmp_path):
        path = tmp_path / "unicode.yaml"
        path.write_text("greeting: 'ciao mondo'\n", encoding="utf-8")
        config = YamlConfig(str(path))
        assert config.get("greeting") == "ciao mondo"

    def test_boolean_values(self, tmp_path):
        path = tmp_path / "booleans.yaml"
        path.write_text("enabled: true\ndisabled: false\n", encoding="utf-8")
        config = YamlConfig(str(path))
        assert config.get("enabled") is True
        assert config.get("disabled") is False

    def test_null_value(self, tmp_path):
        path = tmp_path / "nullval.yaml"
        path.write_text("key: null\n", encoding="utf-8")
        config = YamlConfig(str(path))
        assert config.get("key") is None

    def test_multiline_string(self, tmp_path):
        path = tmp_path / "multiline.yaml"
        path.write_text("prompt: |\n  line one\n  line two\n", encoding="utf-8")
        config = YamlConfig(str(path))
        result = config.get("prompt")
        assert "line one" in result
        assert "line two" in result
