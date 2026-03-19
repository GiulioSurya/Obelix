"""Tests for obelix.core.agent.hooks module.

Covers Hook fluent API, condition evaluation (sync/async), handle() chaining,
effects execution, value transformation, and Outcome construction.
"""

from unittest.mock import MagicMock

import pytest

from obelix.core.agent.hooks import AgentEvent, AgentStatus, Hook, HookDecision, Outcome

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_status(event: AgentEvent = AgentEvent.BEFORE_LLM_CALL, **kwargs):
    """Build a minimal AgentStatus with a mock agent."""
    mock_agent = MagicMock()
    mock_agent.conversation_history = []
    defaults = {"event": event, "agent": mock_agent, "iteration": 1}
    defaults.update(kwargs)
    return AgentStatus(**defaults)


# ---------------------------------------------------------------------------
# AgentEvent
# ---------------------------------------------------------------------------


class TestAgentEvent:
    """Tests for AgentEvent enum values and membership."""

    def test_all_events_present(self):
        """All expected lifecycle events are defined."""
        expected = {
            "BEFORE_LLM_CALL",
            "AFTER_LLM_CALL",
            "BEFORE_TOOL_EXECUTION",
            "AFTER_TOOL_EXECUTION",
            "ON_TOOL_ERROR",
            "BEFORE_FINAL_RESPONSE",
            "QUERY_END",
        }
        actual = {e.name for e in AgentEvent}
        assert actual == expected

    def test_event_is_str_enum(self):
        """AgentEvent members are strings."""
        assert isinstance(AgentEvent.BEFORE_LLM_CALL, str)
        assert AgentEvent.BEFORE_LLM_CALL == "before_llm_call"

    def test_event_count(self):
        """Exactly 7 events are defined."""
        assert len(AgentEvent) == 7


# ---------------------------------------------------------------------------
# HookDecision
# ---------------------------------------------------------------------------


class TestHookDecision:
    """Tests for HookDecision enum."""

    def test_all_decisions_present(self):
        """CONTINUE, RETRY, FAIL, STOP, REJECT are all defined."""
        expected = {"CONTINUE", "RETRY", "FAIL", "STOP", "REJECT"}
        actual = {d.name for d in HookDecision}
        assert actual == expected

    def test_decision_is_str_enum(self):
        """HookDecision members are strings."""
        assert isinstance(HookDecision.CONTINUE, str)
        assert HookDecision.CONTINUE == "continue"


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------


class TestOutcome:
    """Tests for Outcome dataclass."""

    def test_construction(self):
        """Outcome stores decision and value."""
        outcome = Outcome(decision=HookDecision.CONTINUE, value=42)
        assert outcome.decision == HookDecision.CONTINUE
        assert outcome.value == 42

    def test_none_value(self):
        """Outcome can hold None value."""
        outcome = Outcome(decision=HookDecision.STOP, value=None)
        assert outcome.value is None


# ---------------------------------------------------------------------------
# AgentStatus
# ---------------------------------------------------------------------------


class TestAgentStatus:
    """Tests for AgentStatus dataclass."""

    def test_defaults(self):
        """Default optional fields are None/0."""
        status = _make_agent_status()
        assert status.iteration == 1
        assert status.tool_call is None
        assert status.tool_result is None
        assert status.assistant_message is None
        assert status.error is None

    def test_conversation_history_delegates_to_agent(self):
        """conversation_history property returns agent.conversation_history."""
        mock_agent = MagicMock()
        mock_agent.conversation_history = ["msg1", "msg2"]
        status = AgentStatus(event=AgentEvent.BEFORE_LLM_CALL, agent=mock_agent)
        assert status.conversation_history == ["msg1", "msg2"]

    def test_all_fields_set(self):
        """All optional fields can be set explicitly."""
        mock_agent = MagicMock()
        mock_agent.conversation_history = []
        tc = MagicMock()
        tr = MagicMock()
        am = MagicMock()
        status = AgentStatus(
            event=AgentEvent.AFTER_LLM_CALL,
            agent=mock_agent,
            iteration=5,
            tool_call=tc,
            tool_result=tr,
            assistant_message=am,
            error="something broke",
        )
        assert status.iteration == 5
        assert status.tool_call is tc
        assert status.tool_result is tr
        assert status.assistant_message is am
        assert status.error == "something broke"


# ---------------------------------------------------------------------------
# Hook: basic construction and execute
# ---------------------------------------------------------------------------


class TestHookConstruction:
    """Tests for Hook initialization and default behavior."""

    def test_default_decision_is_continue(self):
        """A new Hook defaults to CONTINUE decision."""
        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        assert hook._decision == HookDecision.CONTINUE

    def test_no_condition_no_effects(self):
        """A new Hook has no condition and empty effects."""
        hook = Hook(AgentEvent.AFTER_LLM_CALL)
        assert hook._condition is None
        assert hook._effects == []
        assert hook._value is None

    @pytest.mark.asyncio
    async def test_execute_without_handle_returns_continue_with_original_value(self):
        """execute() without handle() returns CONTINUE with original value."""
        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        status = _make_agent_status()
        outcome = await hook.execute(status, current_value="original")
        assert outcome.decision == HookDecision.CONTINUE
        assert outcome.value == "original"


# ---------------------------------------------------------------------------
# Hook: when() conditions
# ---------------------------------------------------------------------------


class TestHookConditions:
    """Tests for Hook.when() with sync and async conditions."""

    @pytest.mark.asyncio
    async def test_no_condition_always_executes(self):
        """Hook without when() always activates."""
        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.handle(HookDecision.STOP, value="stopped")
        status = _make_agent_status()
        outcome = await hook.execute(status, current_value=None)
        assert outcome.decision == HookDecision.STOP
        assert outcome.value == "stopped"

    @pytest.mark.asyncio
    async def test_sync_condition_true(self):
        """Sync condition returning True activates the hook."""
        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.when(lambda s: True).handle(HookDecision.FAIL)
        status = _make_agent_status()
        outcome = await hook.execute(status, current_value=None)
        assert outcome.decision == HookDecision.FAIL

    @pytest.mark.asyncio
    async def test_sync_condition_false(self):
        """Sync condition returning False skips the hook, returns CONTINUE."""
        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.when(lambda s: False).handle(HookDecision.FAIL)
        status = _make_agent_status()
        outcome = await hook.execute(status, current_value="keep")
        assert outcome.decision == HookDecision.CONTINUE
        assert outcome.value == "keep"

    @pytest.mark.asyncio
    async def test_async_condition_true(self):
        """Async condition returning True activates the hook."""

        async def async_cond(s):
            return True

        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.when(async_cond).handle(HookDecision.RETRY)
        status = _make_agent_status()
        outcome = await hook.execute(status, current_value=None)
        assert outcome.decision == HookDecision.RETRY

    @pytest.mark.asyncio
    async def test_async_condition_false(self):
        """Async condition returning False skips the hook."""

        async def async_cond(s):
            return False

        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.when(async_cond).handle(HookDecision.FAIL)
        status = _make_agent_status()
        outcome = await hook.execute(status, current_value="original")
        assert outcome.decision == HookDecision.CONTINUE
        assert outcome.value == "original"

    @pytest.mark.asyncio
    async def test_condition_receives_agent_status(self):
        """Condition callable receives the AgentStatus object."""
        received = {}

        def capture_status(s):
            received["status"] = s
            return True

        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.when(capture_status).handle(HookDecision.CONTINUE)
        status = _make_agent_status(iteration=7)
        await hook.execute(status)
        assert received["status"] is status
        assert received["status"].iteration == 7


# ---------------------------------------------------------------------------
# Hook: handle() with value transformation
# ---------------------------------------------------------------------------


class TestHookValueTransformation:
    """Tests for Hook.handle() value parameter (direct and callable)."""

    @pytest.mark.asyncio
    async def test_direct_value(self):
        """handle(value=<non-callable>) sets the outcome value directly."""
        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.handle(HookDecision.STOP, value="direct_value")
        status = _make_agent_status()
        outcome = await hook.execute(status, current_value=None)
        assert outcome.value == "direct_value"

    @pytest.mark.asyncio
    async def test_callable_sync_value(self):
        """handle(value=<sync callable>) calls it with (status, current_value)."""
        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.handle(
            HookDecision.CONTINUE,
            value=lambda s, cv: f"transformed_{cv}",
        )
        status = _make_agent_status()
        outcome = await hook.execute(status, current_value="input")
        assert outcome.value == "transformed_input"

    @pytest.mark.asyncio
    async def test_callable_async_value(self):
        """handle(value=<async callable>) awaits the coroutine."""

        async def async_value(s, cv):
            return f"async_{cv}"

        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.handle(HookDecision.CONTINUE, value=async_value)
        status = _make_agent_status()
        outcome = await hook.execute(status, current_value="data")
        assert outcome.value == "async_data"

    @pytest.mark.asyncio
    async def test_none_value_keeps_current(self):
        """handle(value=None) preserves the current_value."""
        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.handle(HookDecision.CONTINUE, value=None)
        status = _make_agent_status()
        outcome = await hook.execute(status, current_value="preserved")
        assert outcome.value == "preserved"


# ---------------------------------------------------------------------------
# Hook: effects
# ---------------------------------------------------------------------------


class TestHookEffects:
    """Tests for Hook.handle() effects execution."""

    @pytest.mark.asyncio
    async def test_sync_effect_executed(self):
        """Sync effect function is called during execute()."""
        called = []
        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.handle(HookDecision.CONTINUE, effects=[lambda s: called.append(True)])
        status = _make_agent_status()
        await hook.execute(status)
        assert called == [True]

    @pytest.mark.asyncio
    async def test_async_effect_executed(self):
        """Async effect function is awaited during execute()."""
        called = []

        async def async_effect(s):
            called.append("async")

        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.handle(HookDecision.CONTINUE, effects=[async_effect])
        status = _make_agent_status()
        await hook.execute(status)
        assert called == ["async"]

    @pytest.mark.asyncio
    async def test_multiple_effects_executed_in_order(self):
        """Multiple effects are executed in the order they are provided."""
        order = []
        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.handle(
            HookDecision.CONTINUE,
            effects=[
                lambda s: order.append("first"),
                lambda s: order.append("second"),
                lambda s: order.append("third"),
            ],
        )
        status = _make_agent_status()
        await hook.execute(status)
        assert order == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_effects_run_before_value_computation(self):
        """Effects run before value is computed."""
        state = {"counter": 0}

        def increment_effect(s):
            state["counter"] += 1

        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.handle(
            HookDecision.CONTINUE,
            value=lambda s, cv: state["counter"],
            effects=[increment_effect],
        )
        status = _make_agent_status()
        outcome = await hook.execute(status, current_value=None)
        assert outcome.value == 1


# ---------------------------------------------------------------------------
# Hook: chaining
# ---------------------------------------------------------------------------


class TestHookChaining:
    """Tests for when().handle() fluent chaining."""

    def test_when_returns_self(self):
        """when() returns the Hook for chaining."""
        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        result = hook.when(lambda s: True)
        assert result is hook

    def test_handle_returns_self(self):
        """handle() returns the Hook for chaining."""
        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        result = hook.handle(HookDecision.CONTINUE)
        assert result is hook

    @pytest.mark.asyncio
    async def test_full_chain(self):
        """when().handle() chain produces correct Outcome."""
        hook = Hook(AgentEvent.BEFORE_LLM_CALL)
        hook.when(lambda s: s.iteration > 2).handle(
            HookDecision.FAIL, value="too many iterations"
        )
        status_iter1 = _make_agent_status(iteration=1)
        outcome1 = await hook.execute(status_iter1)
        assert outcome1.decision == HookDecision.CONTINUE

        status_iter3 = _make_agent_status(iteration=3)
        outcome3 = await hook.execute(status_iter3)
        assert outcome3.decision == HookDecision.FAIL
        assert outcome3.value == "too many iterations"
