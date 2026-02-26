"""Tests for obelix.core.agent.subagent_wrapper.SubAgentWrapper.

Covers initialization, field extraction, schema building, query building,
stateless execution, stateful execution, and MCP schema generation.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import Field, ValidationError

from obelix.core.agent.base_agent import BaseAgent
from obelix.core.agent.subagent_wrapper import SubAgentWrapper
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.tool_message import (
    MCPToolSchema,
    ToolCall,
    ToolResult,
    ToolStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleAgent(BaseAgent):
    """Agent subclass with no extra fields for basic tests."""

    pass


class _AgentWithFields(BaseAgent):
    """Agent subclass with annotated Pydantic Fields for field extraction tests."""

    language: str = Field(default="en", description="Response language")
    style: str = Field(default="formal", description="Response style")


class _AgentWithMixed(BaseAgent):
    """Agent with a mix of Field, plain class-level attr, and private attr."""

    context: str = Field(default="", description="Additional context")
    _private_field: str = "hidden"
    plain_attr: str = "not_a_field"  # no FieldInfo, should be ignored


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSubAgentWrapperConstruction:
    """Tests for SubAgentWrapper.__init__ and basic attributes."""

    def test_tool_name_set(self, mock_provider):
        """tool_name matches the name parameter."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="helper", description="Helps")
        assert wrapper.tool_name == "helper"

    def test_tool_description_set(self, mock_provider):
        """tool_description matches the description parameter."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="helper", description="Helps a lot")
        assert wrapper.tool_description == "Helps a lot"

    def test_stateless_default_false(self, mock_provider):
        """stateless defaults to False."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d")
        assert wrapper._stateless is False

    def test_stateless_true(self, mock_provider):
        """stateless=True is stored correctly."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d", stateless=True)
        assert wrapper._stateless is True

    def test_agent_stored(self, mock_provider):
        """_agent attribute holds the wrapped agent."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d")
        assert wrapper._agent is agent

    def test_execution_lock_created(self, mock_provider):
        """An asyncio.Lock is created for stateful execution."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d")
        assert isinstance(wrapper._execution_lock, asyncio.Lock)


# ---------------------------------------------------------------------------
# _extract_fields
# ---------------------------------------------------------------------------


class TestExtractFields:
    """Tests for SubAgentWrapper._extract_fields static method."""

    def test_always_includes_query(self, mock_provider):
        """query field is always present as the first key."""
        fields = SubAgentWrapper._extract_fields(_SimpleAgent)
        assert "query" in fields
        keys = list(fields.keys())
        assert keys[0] == "query"

    def test_pydantic_fields_extracted(self, mock_provider):
        """Fields with FieldInfo annotations are extracted."""
        fields = SubAgentWrapper._extract_fields(_AgentWithFields)
        assert "language" in fields
        assert "style" in fields

    def test_non_field_ignored(self, mock_provider):
        """Class attributes without FieldInfo are ignored."""
        fields = SubAgentWrapper._extract_fields(_AgentWithMixed)
        assert "plain_attr" not in fields

    def test_private_attrs_ignored(self, mock_provider):
        """Attributes starting with _ are ignored."""
        fields = SubAgentWrapper._extract_fields(_AgentWithMixed)
        assert "_private_field" not in fields

    def test_query_not_duplicated(self, mock_provider):
        """If the agent class had a 'query' annotation, it is skipped (not duplicated)."""
        fields = SubAgentWrapper._extract_fields(_AgentWithFields)
        # query should appear exactly once
        keys = list(fields.keys())
        assert keys.count("query") == 1

    def test_context_field_extracted(self, mock_provider):
        """The context field (FieldInfo) from _AgentWithMixed is extracted."""
        fields = SubAgentWrapper._extract_fields(_AgentWithMixed)
        assert "context" in fields


# ---------------------------------------------------------------------------
# _build_input_schema
# ---------------------------------------------------------------------------


class TestBuildInputSchema:
    """Tests for SubAgentWrapper._build_input_schema."""

    def test_creates_pydantic_model(self, mock_provider):
        """Returns a Pydantic model class."""
        from pydantic import BaseModel

        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="test_agent", description="d")
        assert issubclass(wrapper._input_schema, BaseModel)

    def test_query_is_required(self, mock_provider):
        """The query field is required in the schema."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="test_agent", description="d")
        schema = wrapper._input_schema.model_json_schema()
        assert "query" in schema.get("required", [])

    def test_schema_class_name_based_on_tool_name(self, mock_provider):
        """Schema class name is derived from the tool name."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="my_helper", description="d")
        assert "MyHelper" in wrapper._input_schema.__name__

    def test_extra_fields_in_schema(self, mock_provider):
        """Agent-specific fields appear in the schema properties."""
        agent = _AgentWithFields(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="poly", description="d")
        schema = wrapper._input_schema.model_json_schema()
        assert "language" in schema.get("properties", {})
        assert "style" in schema.get("properties", {})


# ---------------------------------------------------------------------------
# _build_query
# ---------------------------------------------------------------------------


class TestBuildQuery:
    """Tests for SubAgentWrapper._build_query."""

    def test_simple_query_only(self, mock_provider):
        """With no extra fields, query text is returned as-is."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d")
        tc = ToolCall(id="tc_1", name="w", arguments={"query": "What is 2+2?"})
        result = wrapper._build_query(tc)
        assert result == "What is 2+2?"

    def test_context_fields_prepended(self, mock_provider):
        """Extra fields are prepended to the query text."""
        agent = _AgentWithFields(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d")
        tc = ToolCall(
            id="tc_1",
            name="w",
            arguments={"query": "Translate this", "language": "it", "style": "casual"},
        )
        result = wrapper._build_query(tc)
        assert "language: it" in result
        assert "style: casual" in result
        assert result.endswith("Translate this")

    def test_empty_context_fields_omitted(self, mock_provider):
        """Empty/falsy context fields are not included."""
        agent = _AgentWithFields(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d")
        tc = ToolCall(
            id="tc_1",
            name="w",
            arguments={"query": "Hello", "language": "", "style": ""},
        )
        result = wrapper._build_query(tc)
        # Only the query should be present
        assert result == "Hello"

    def test_validates_input_against_schema(self, mock_provider):
        """Invalid input (missing required query) raises validation error."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d")
        tc = ToolCall(id="tc_1", name="w", arguments={})
        with pytest.raises(ValidationError):
            wrapper._build_query(tc)


# ---------------------------------------------------------------------------
# execute (stateless)
# ---------------------------------------------------------------------------


class TestStatelessExecution:
    """Tests for SubAgentWrapper stateless execution."""

    @pytest.mark.asyncio
    async def test_stateless_returns_success_result(self, mock_provider):
        """Stateless execute returns ToolResult with SUCCESS on normal execution."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d", stateless=True)
        tc = ToolCall(id="tc_1", name="w", arguments={"query": "hello"})
        result = await wrapper.execute(tc)
        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.SUCCESS
        assert result.tool_name == "w"
        assert result.tool_call_id == "tc_1"

    @pytest.mark.asyncio
    async def test_stateless_uses_copy(self, mock_provider):
        """Stateless execution operates on a copy, original agent history unchanged."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        original_history_len = len(agent.conversation_history)
        wrapper = SubAgentWrapper(agent, name="w", description="d", stateless=True)
        tc = ToolCall(id="tc_1", name="w", arguments={"query": "hello"})
        await wrapper.execute(tc)
        # Original agent history should not grow from the sub-execution
        assert len(agent.conversation_history) == original_history_len

    @pytest.mark.asyncio
    async def test_stateless_exception_returns_error_result(self, mock_provider):
        """Stateless execute captures exceptions as ERROR ToolResult."""
        mock_provider.invoke = AsyncMock(side_effect=RuntimeError("LLM is down"))
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d", stateless=True)
        tc = ToolCall(id="tc_1", name="w", arguments={"query": "boom"})
        result = await wrapper.execute(tc)
        assert result.status == ToolStatus.ERROR
        assert "LLM is down" in result.error

    @pytest.mark.asyncio
    async def test_stateless_result_content_matches_response(self, mock_provider):
        """Stateless result.result contains the AssistantResponse content."""
        mock_provider.invoke = AsyncMock(
            return_value=AssistantMessage(content="answer 42")
        )
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d", stateless=True)
        tc = ToolCall(id="tc_1", name="w", arguments={"query": "meaning of life"})
        result = await wrapper.execute(tc)
        assert result.result == "answer 42"


# ---------------------------------------------------------------------------
# execute (stateful)
# ---------------------------------------------------------------------------


class TestStatefulExecution:
    """Tests for SubAgentWrapper stateful execution."""

    @pytest.mark.asyncio
    async def test_stateful_returns_success_result(self, mock_provider):
        """Stateful execute returns ToolResult with SUCCESS."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d", stateless=False)
        tc = ToolCall(id="tc_1", name="w", arguments={"query": "hello"})
        result = await wrapper.execute(tc)
        assert isinstance(result, ToolResult)
        assert result.status == ToolStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_stateful_uses_original_agent(self, mock_provider):
        """Stateful execution modifies the original agent's conversation history."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        original_history_len = len(agent.conversation_history)
        wrapper = SubAgentWrapper(agent, name="w", description="d", stateless=False)
        tc = ToolCall(id="tc_1", name="w", arguments={"query": "hello"})
        await wrapper.execute(tc)
        # Stateful: original agent history should grow
        assert len(agent.conversation_history) > original_history_len

    @pytest.mark.asyncio
    async def test_stateful_exception_returns_error_result(self, mock_provider):
        """Stateful execute captures exceptions as ERROR ToolResult."""
        mock_provider.invoke = AsyncMock(side_effect=ValueError("bad input"))
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d", stateless=False)
        tc = ToolCall(id="tc_1", name="w", arguments={"query": "fail"})
        result = await wrapper.execute(tc)
        assert result.status == ToolStatus.ERROR
        assert "bad input" in result.error

    @pytest.mark.asyncio
    async def test_stateful_has_execution_time(self, mock_provider):
        """Stateful result includes execution_time."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d", stateless=False)
        tc = ToolCall(id="tc_1", name="w", arguments={"query": "time check"})
        result = await wrapper.execute(tc)
        assert result.execution_time is not None
        assert result.execution_time >= 0


# ---------------------------------------------------------------------------
# create_schema
# ---------------------------------------------------------------------------


class TestCreateSchema:
    """Tests for SubAgentWrapper.create_schema."""

    def test_returns_mcp_tool_schema(self, mock_provider):
        """create_schema returns MCPToolSchema."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="my_tool", description="My tool desc")
        schema = wrapper.create_schema()
        assert isinstance(schema, MCPToolSchema)

    def test_schema_name_matches_tool_name(self, mock_provider):
        """MCPToolSchema.name matches tool_name."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="analyzer", description="Analyzes")
        schema = wrapper.create_schema()
        assert schema.name == "analyzer"

    def test_schema_description_matches(self, mock_provider):
        """MCPToolSchema.description matches tool_description."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="analyzer", description="Analyzes stuff")
        schema = wrapper.create_schema()
        assert schema.description == "Analyzes stuff"

    def test_schema_input_schema_has_query(self, mock_provider):
        """inputSchema includes a query property."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d")
        schema = wrapper.create_schema()
        assert "query" in schema.inputSchema.get("properties", {})

    def test_schema_output_schema_is_object(self, mock_provider):
        """outputSchema is a generic object schema."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d")
        schema = wrapper.create_schema()
        assert schema.outputSchema == {"type": "object", "additionalProperties": True}

    def test_schema_with_extra_fields(self, mock_provider):
        """Extra agent fields appear in inputSchema properties."""
        agent = _AgentWithFields(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d")
        schema = wrapper.create_schema()
        props = schema.inputSchema.get("properties", {})
        assert "language" in props
        assert "style" in props


# ---------------------------------------------------------------------------
# _tracer property
# ---------------------------------------------------------------------------


class TestTracerPropagation:
    """Tests for SubAgentWrapper._tracer property."""

    def test_tracer_propagated_from_agent(self, mock_provider):
        """_tracer returns the wrapped agent's _tracer."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        mock_tracer = MagicMock()
        agent._tracer = mock_tracer
        wrapper = SubAgentWrapper(agent, name="w", description="d")
        assert wrapper._tracer is mock_tracer

    def test_tracer_none_when_agent_has_none(self, mock_provider):
        """_tracer returns None when agent has no tracer."""
        agent = _SimpleAgent(system_message="x", provider=mock_provider)
        wrapper = SubAgentWrapper(agent, name="w", description="d")
        assert wrapper._tracer is None
