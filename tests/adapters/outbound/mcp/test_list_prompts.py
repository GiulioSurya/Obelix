from unittest.mock import MagicMock

import pytest

from obelix.adapters.outbound.mcp.manager import MCPManager, MCPPrompt


def test_list_prompts_returns_empty_when_not_connected():
    manager = MCPManager(config=[])
    assert manager.list_prompts() == []


def test_list_prompts_returns_empty_when_group_missing():
    manager = MCPManager(config=[])
    # Simulate stale state: connected flag but no group
    manager._connected = True
    manager._group = None
    assert manager.list_prompts() == []


def test_list_prompts_single_server():
    manager = MCPManager(config=[])
    prompt = MagicMock()
    prompt.description = "Does a thing"
    prompt.arguments = []

    group = MagicMock()
    # SDK exposes prompts dict keyed by (possibly namespaced) prompt name
    group.prompts = {"mcp__tracer__review": prompt}

    manager._group = group
    manager._connected = True
    # monkey-patch _resolve_server_name to return the namespace we want
    manager._resolve_server_name = MagicMock(return_value="tracer")

    prompts = manager.list_prompts()
    assert len(prompts) == 1
    p = prompts[0]
    assert isinstance(p, MCPPrompt)
    assert p.name == "mcp__tracer__review"
    assert p.server_name == "tracer"
    assert p.description == "Does a thing"


def test_list_prompts_aggregates_multiple():
    manager = MCPManager(config=[])
    p1, p2 = MagicMock(), MagicMock()
    p1.description = "one"
    p1.arguments = []
    p2.description = "two"
    p2.arguments = []
    group = MagicMock()
    group.prompts = {"mcp__a__x": p1, "mcp__b__y": p2}
    manager._group = group
    manager._connected = True
    manager._resolve_server_name = MagicMock(side_effect=lambda n: n.split("__")[1])

    prompts = manager.list_prompts()
    assert len(prompts) == 2
    names = {p.name for p in prompts}
    assert names == {"mcp__a__x", "mcp__b__y"}


def test_list_prompts_handles_missing_prompts_attribute():
    """SDK version without `prompts` attribute should not crash."""
    manager = MCPManager(config=[])

    class NoPromptsGroup:
        pass

    manager._group = NoPromptsGroup()
    manager._connected = True

    assert manager.list_prompts() == []


def test_list_prompts_preserves_arguments():
    manager = MCPManager(config=[])
    arg1 = MagicMock()
    arg1.name = "topic"
    prompt = MagicMock()
    prompt.description = "d"
    prompt.arguments = [arg1]

    group = MagicMock()
    group.prompts = {"p": prompt}
    manager._group = group
    manager._connected = True
    manager._resolve_server_name = MagicMock(return_value="srv")

    prompts = manager.list_prompts()
    assert len(prompts[0].arguments) == 1


def test_mcp_prompt_is_frozen_dataclass():
    import dataclasses
    from dataclasses import is_dataclass

    assert is_dataclass(MCPPrompt)
    # Check it's frozen
    p = MCPPrompt(name="n", description="d", arguments=[], server_name="s")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.name = "x"


def test_list_prompts_handles_description_none():
    """SDK may return a Prompt with description=None; coerce to empty string."""
    manager = MCPManager(config=[])
    prompt = MagicMock()
    prompt.description = None
    prompt.arguments = []
    group = MagicMock()
    group.prompts = {"p": prompt}
    manager._group = group
    manager._connected = True
    manager._resolve_server_name = MagicMock(return_value="s")

    prompts = manager.list_prompts()
    assert len(prompts) == 1
    assert prompts[0].description == ""


def test_list_prompts_handles_arguments_none():
    """SDK may return a Prompt with arguments=None; coerce to empty list."""
    manager = MCPManager(config=[])
    prompt = MagicMock()
    prompt.description = "d"
    prompt.arguments = None
    group = MagicMock()
    group.prompts = {"p": prompt}
    manager._group = group
    manager._connected = True
    manager._resolve_server_name = MagicMock(return_value="s")

    prompts = manager.list_prompts()
    assert len(prompts) == 1
    assert prompts[0].arguments == []


def test_list_prompts_empty_dict():
    """An empty prompts dict returns an empty list (distinct from missing attr)."""
    manager = MCPManager(config=[])
    group = MagicMock()
    group.prompts = {}
    manager._group = group
    manager._connected = True
    assert manager.list_prompts() == []
