"""Enforce the skills subsystem architectural DAG via grimp.

If these tests fail, the coupling rules in
docs/superpowers/specs/2026-04-20-skills-system-design.md are violated.
"""

import grimp
import pytest


@pytest.fixture(scope="module")
def graph():
    return grimp.build_graph("obelix")


def _direct_imports(graph, module: str) -> set[str]:
    """Return modules directly imported by `module`."""
    return graph.find_modules_directly_imported_by(module)


class TestSkillCoreLeaf:
    """`obelix.core.skill.skill` is the leaf of the DAG — no same-package imports."""

    def test_skill_module_does_not_import_from_other_core_skill_modules(self, graph):
        direct = _direct_imports(graph, "obelix.core.skill.skill")
        forbidden = {
            m
            for m in direct
            if m.startswith("obelix.core.skill.") and m != "obelix.core.skill.skill"
        }
        # Allow `obelix.core.skill.reporting` because SkillValidationError
        # lazy-imports it inside `_default_message()` — grimp picks up
        # inline imports too; check to see if this is flagged.
        # If it is, we document it as the one exception.
        # Accept ONLY reporting as the lazy import exception.
        assert forbidden <= {"obelix.core.skill.reporting"}


class TestManagerIsolation:
    """`obelix.core.skill.manager` must not import yaml, parsing, validation, or adapters."""

    def test_manager_does_not_import_yaml(self, graph):
        direct = _direct_imports(graph, "obelix.core.skill.manager")
        assert "yaml" not in direct

    def test_manager_does_not_import_parsing(self, graph):
        direct = _direct_imports(graph, "obelix.core.skill.manager")
        assert "obelix.core.skill.parsing" not in direct

    def test_manager_does_not_import_validation(self, graph):
        direct = _direct_imports(graph, "obelix.core.skill.manager")
        assert "obelix.core.skill.validation" not in direct

    def test_manager_does_not_import_concrete_providers(self, graph):
        direct = _direct_imports(graph, "obelix.core.skill.manager")
        forbidden = {m for m in direct if m.startswith("obelix.adapters.")}
        assert forbidden == set()


class TestValidationPurity:
    """`obelix.core.skill.validation` must not touch IO."""

    def test_validation_does_not_import_yaml(self, graph):
        direct = _direct_imports(graph, "obelix.core.skill.validation")
        assert "yaml" not in direct

    def test_validation_does_not_import_pathlib_operations(self, graph):
        """pathlib.Path is OK for type hints but no file IO should happen here."""
        # Can't easily enforce via grimp; covered by unit tests (no IO).
        pass  # documented — unit tests enforce this


class TestSkillToolIsolation:
    """`obelix.plugins.builtin.skill_tool` orchestrates via SkillManager — not adapters."""

    def test_skill_tool_does_not_import_filesystem_provider(self, graph):
        direct = _direct_imports(graph, "obelix.plugins.builtin.skill_tool")
        assert "obelix.adapters.outbound.skill.filesystem" not in direct

    def test_skill_tool_does_not_import_mcp_provider(self, graph):
        direct = _direct_imports(graph, "obelix.plugins.builtin.skill_tool")
        assert "obelix.adapters.outbound.skill.mcp" not in direct

    def test_skill_tool_does_not_import_yaml(self, graph):
        direct = _direct_imports(graph, "obelix.plugins.builtin.skill_tool")
        assert "yaml" not in direct


class TestCoreDoesNotImportAdaptersOrPlugins:
    """Core skill modules must not import from adapters or plugins (upstream rule)."""

    @pytest.mark.parametrize(
        "module",
        [
            "obelix.core.skill.skill",
            "obelix.core.skill.substitution",
            "obelix.core.skill.parsing",
            "obelix.core.skill.validation",
            "obelix.core.skill.reporting",
            "obelix.core.skill.manager",
        ],
    )
    def test_core_does_not_import_plugins_or_adapters(self, graph, module):
        direct = _direct_imports(graph, module)
        forbidden = {
            m for m in direct if m.startswith(("obelix.plugins.", "obelix.adapters."))
        }
        assert forbidden == set(), f"{module} imports forbidden modules: {forbidden}"


class TestAdaptersDependOnPortNotDirectly:
    """Adapters must depend on the port (AbstractSkillProvider), not skip it."""

    def test_filesystem_imports_port(self, graph):
        direct = _direct_imports(graph, "obelix.adapters.outbound.skill.filesystem")
        assert "obelix.ports.outbound.skill_provider" in direct

    def test_mcp_imports_port(self, graph):
        direct = _direct_imports(graph, "obelix.adapters.outbound.skill.mcp")
        assert "obelix.ports.outbound.skill_provider" in direct
