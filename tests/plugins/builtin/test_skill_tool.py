import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from obelix.core.skill.manager import SkillManager
from obelix.core.skill.skill import Skill
from obelix.plugins.builtin.skill_tool import DEFAULT_LISTING_BUDGET, make_skill_tool


def _mgr_with(*skills: Skill) -> SkillManager:
    provider = MagicMock()
    provider.discover.return_value = list(skills)
    return SkillManager(providers=[provider])


def _skill(
    name: str = "s",
    desc: str = "d",
    when: str | None = None,
    body: str = "body",
) -> Skill:
    return Skill(
        name=name,
        description=desc,
        body=body,
        base_dir=Path("/tmp/skills") / name,
        when_to_use=when,
    )


class TestSystemPromptFragment:
    def test_no_skills_empty_string(self):
        skill_tool = make_skill_tool(_mgr_with())
        assert skill_tool.system_prompt_fragment() == ""

    def test_with_skills_lists_all(self):
        mgr = _mgr_with(_skill("a", "desc a"), _skill("b", "desc b"))
        skill_tool = make_skill_tool(mgr)
        frag = skill_tool.system_prompt_fragment()
        assert "a" in frag
        assert "b" in frag
        assert "desc a" in frag
        assert "desc b" in frag

    def test_fragment_mentions_skill_tool_invocation(self):
        mgr = _mgr_with(_skill("a"))
        skill_tool = make_skill_tool(mgr)
        frag = skill_tool.system_prompt_fragment()
        assert "Skill" in frag  # references the tool name
        assert "Available skills" in frag

    def test_fragment_includes_when_to_use(self):
        mgr = _mgr_with(_skill("a", "desc", when="user asks X"))
        skill_tool = make_skill_tool(mgr)
        frag = skill_tool.system_prompt_fragment()
        assert "user asks X" in frag

    def test_fragment_tells_llm_when_to_use_tool(self):
        mgr = _mgr_with(_skill("a"))
        skill_tool = make_skill_tool(mgr)
        frag = skill_tool.system_prompt_fragment()
        # Nudges the model toward proactive invocation
        assert "invoke" in frag.lower() or "use" in frag.lower()

    def test_custom_budget_enforces_tight_bound(self):
        """Budget truncation must hold: fragment size stays close to budget."""
        long_desc = "x" * 5000
        mgr = _mgr_with(_skill("a", long_desc))
        skill_tool = make_skill_tool(mgr, listing_budget=200)
        frag = skill_tool.system_prompt_fragment()
        # The listing itself is budgeted; the surrounding header adds a small
        # constant. Assert the fragment stays within a tight bound (budget +
        # header overhead) — not just "less than the raw description".
        assert len(frag) < 400, f"expected tight budget, got {len(frag)} chars"

    def test_custom_budget_truncates_long_description(self):
        """Long descriptions should not appear in full when budget is small."""
        long_desc = "x" * 5000
        mgr = _mgr_with(_skill("a", long_desc))
        skill_tool = make_skill_tool(mgr, listing_budget=200)
        frag = skill_tool.system_prompt_fragment()
        # The full 5000-char string must NOT be present — must be truncated.
        assert long_desc not in frag


# TestExecuteStub deferred to Task 7.2: the @tool decorator wraps execute()
# to accept a ToolCall object and returns a coroutine producing ToolResult.
# Stubbing that contract for the scaffold is premature — Task 7.2 will
# introduce the real invocation logic (substitution + hooks) and tests.


class TestToolExposure:
    def test_has_tool_name(self):
        skill_tool = make_skill_tool(_mgr_with(_skill("a")))
        assert getattr(skill_tool, "tool_name", None) == "Skill"

    def test_has_tool_description(self):
        skill_tool = make_skill_tool(_mgr_with(_skill("a")))
        desc = getattr(skill_tool, "tool_description", "")
        assert desc  # non-empty

    def test_default_budget_is_8000_chars(self):
        assert DEFAULT_LISTING_BUDGET == 8_000

    def test_session_id_accepted_but_not_stored(self):
        """session_id is accepted for forward compat (Task 7.2+) but not used yet."""
        mgr = _mgr_with(_skill("a"))
        # Passing a session_id must not raise
        skill_tool = make_skill_tool(mgr, session_id="abc-123")
        # The tool instance should still work normally
        assert skill_tool.system_prompt_fragment() != ""


def _invoke_execute(skill_tool_instance, **fields):
    """Execute the underlying @tool-wrapped class with given fields.

    The @tool decorator wraps execute to accept a ToolCall and returns a
    coroutine producing ToolResult. `asyncio.run` drives the coroutine
    without relying on the deprecated global event loop.
    """
    from obelix.core.model.tool_message import ToolCall

    cls = type(skill_tool_instance)
    instance = cls()
    call = ToolCall(id="test-1", name="Skill", arguments=fields)
    return asyncio.run(instance.execute(call))


class TestInlineInvocation:
    def test_unknown_skill_returns_error_result(self):
        mgr = _mgr_with(_skill("alpha"), _skill("beta"))
        skill_tool = make_skill_tool(mgr)
        result = _invoke_execute(skill_tool, name="missing", args="")
        from obelix.core.model.tool_message import ToolStatus

        assert result.status == ToolStatus.ERROR
        err_text = (result.error or "") + str(result.result or "")
        assert "missing" in err_text
        assert "alpha" in err_text
        assert "beta" in err_text

    def test_known_skill_returns_body_as_success(self):
        mgr = _mgr_with(_skill("alpha", body="Alpha body text"))
        skill_tool = make_skill_tool(mgr)
        result = _invoke_execute(skill_tool, name="alpha", args="")
        from obelix.core.model.tool_message import ToolStatus

        assert result.status == ToolStatus.SUCCESS
        assert result.result == "Alpha body text"

    def test_known_skill_with_flat_arguments(self):
        mgr = _mgr_with(_skill("s", body="Got: $ARGUMENTS"))
        skill_tool = make_skill_tool(mgr)
        result = _invoke_execute(skill_tool, name="s", args="hello world")
        assert result.result == "Got: hello world"

    def test_known_skill_with_named_positional_args(self):
        real_skill = Skill(
            name="s",
            description="d",
            body="path=$path depth=$depth",
            base_dir=None,
            arguments=("path", "depth"),
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        skill_tool = make_skill_tool(mgr)
        result = _invoke_execute(skill_tool, name="s", args="foo.py 3")
        assert result.result == "path=foo.py depth=3"

    def test_args_overflow_returns_error(self):
        real_skill = Skill(
            name="s",
            description="d",
            body="x=$x",
            base_dir=None,
            arguments=("x",),
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        skill_tool = make_skill_tool(mgr)
        result = _invoke_execute(skill_tool, name="s", args="a b c")
        from obelix.core.model.tool_message import ToolStatus

        assert result.status == ToolStatus.ERROR
        err_text = (result.error or "") + str(result.result or "")
        assert "expects" in err_text or "got" in err_text

    def test_obelix_skill_dir_substituted(self):
        real_skill = Skill(
            name="s",
            description="d",
            body="dir=${OBELIX_SKILL_DIR}",
            base_dir=Path("/abs/skills/s"),
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        skill_tool = make_skill_tool(mgr)
        result = _invoke_execute(skill_tool, name="s", args="")
        assert "${OBELIX_SKILL_DIR}" not in (result.result or "")
        assert "s" in (result.result or "")

    def test_session_id_substituted_when_provided(self):
        real_skill = Skill(
            name="s",
            description="d",
            body="sid=${OBELIX_SESSION_ID}",
            base_dir=None,
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        skill_tool = make_skill_tool(mgr, session_id="session-42")
        result = _invoke_execute(skill_tool, name="s", args="")
        assert result.result == "sid=session-42"

    def test_session_id_default_uuid_when_not_provided(self):
        """When no session_id provided, a stable UUID is used for the tool instance."""
        import uuid as uuid_lib

        # Two distinct skills so idempotence (Task 7.3) does not short-circuit
        # the second call; both share the tool instance's session id.
        skill_a = Skill(
            name="s1",
            description="d",
            body="sid=${OBELIX_SESSION_ID}",
            base_dir=None,
        )
        skill_b = Skill(
            name="s2",
            description="d",
            body="sid=${OBELIX_SESSION_ID}",
            base_dir=None,
        )
        provider = MagicMock()
        provider.discover.return_value = [skill_a, skill_b]
        mgr = SkillManager(providers=[provider])
        skill_tool = make_skill_tool(mgr)  # no session_id
        result1 = _invoke_execute(skill_tool, name="s1", args="")
        result2 = _invoke_execute(skill_tool, name="s2", args="")
        # Same tool instance -> same session id across calls
        assert result1.result == result2.result
        # The payload is exactly "sid=" + a valid UUID string (36 chars)
        assert result1.result.startswith("sid=")
        uuid_part = result1.result[len("sid=") :]
        # Raises ValueError if not a valid UUID — stronger than length check
        uuid_lib.UUID(uuid_part)
        assert len(uuid_part) == 36

    def test_unknown_skill_with_empty_manager_no_trailing_available(self):
        """Empty manager produces a clean error message, not 'Available: '."""
        mgr = _mgr_with()  # zero skills
        skill_tool = make_skill_tool(mgr)
        result = _invoke_execute(skill_tool, name="anything", args="")
        from obelix.core.model.tool_message import ToolStatus

        assert result.status == ToolStatus.ERROR
        err_text = (result.error or "") + str(result.result or "")
        assert "No skills registered" in err_text
        assert "Available: " not in err_text  # no dangling trailer

    def test_substitution_error_includes_skill_name(self):
        """Args-overflow error mentions WHICH skill failed."""
        real_skill = Skill(
            name="my-skill",
            description="d",
            body="x=$x",
            base_dir=None,
            arguments=("x",),
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        skill_tool = make_skill_tool(mgr)
        result = _invoke_execute(skill_tool, name="my-skill", args="a b c")
        err_text = (result.error or "") + str(result.result or "")
        assert "my-skill" in err_text


class TestIdempotence:
    """A skill invoked twice in the same query returns 'already active' the second time."""

    def test_second_invocation_returns_already_active(self):
        mgr = _mgr_with(_skill("alpha", body="Body of alpha"))
        skill_tool = make_skill_tool(mgr)
        # First invocation: full body
        r1 = _invoke_execute(skill_tool, name="alpha", args="")
        assert r1.result == "Body of alpha"
        # Second invocation of SAME tool instance, SAME skill: idempotent
        r2 = _invoke_execute(skill_tool, name="alpha", args="")
        from obelix.core.model.tool_message import ToolStatus

        assert r2.status == ToolStatus.SUCCESS
        assert "already active" in (r2.result or "").lower()

    def test_different_skills_not_affected(self):
        mgr = _mgr_with(
            _skill("alpha", body="Body A"),
            _skill("beta", body="Body B"),
        )
        skill_tool = make_skill_tool(mgr)
        r1 = _invoke_execute(skill_tool, name="alpha", args="")
        r2 = _invoke_execute(skill_tool, name="beta", args="")
        assert r1.result == "Body A"
        assert r2.result == "Body B"  # beta not affected by alpha being active

    def test_idempotence_is_per_tool_instance(self):
        mgr = _mgr_with(_skill("alpha", body="Body A"))
        tool1 = make_skill_tool(mgr)
        _invoke_execute(tool1, name="alpha", args="")
        # Fresh tool instance has fresh active-set
        tool2 = make_skill_tool(mgr)
        r2 = _invoke_execute(tool2, name="alpha", args="")
        assert r2.result == "Body A"


class TestHookRegistration:
    """Skill frontmatter hooks are registered on the parent agent at first invoke."""

    def test_skill_without_hooks_no_registration(self):
        mgr = _mgr_with(_skill("alpha", body="Body"))
        parent = MagicMock()
        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        _invoke_execute(skill_tool, name="alpha", args="")
        # parent.on(...) never called
        assert not parent.on.called

    def test_skill_with_hooks_registers_on_parent(self):
        real_skill = Skill(
            name="alpha",
            description="d",
            body="Body",
            base_dir=None,
            hooks={"on_tool_error": "Retry carefully"},
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        parent = MagicMock()
        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        _invoke_execute(skill_tool, name="alpha", args="")
        # parent.on(AgentEvent.ON_TOOL_ERROR) was called
        from obelix.core.agent.hooks import AgentEvent, HookDecision

        parent.on.assert_any_call(AgentEvent.ON_TOOL_ERROR)
        # And .handle(CONTINUE, effects=[...]) was called on the returned Hook
        returned_hook = parent.on.return_value
        assert returned_hook.handle.called
        # Verify the decision passed is CONTINUE and effects list is non-empty
        call_args = returned_hook.handle.call_args
        assert call_args.args[0] == HookDecision.CONTINUE
        effects = call_args.kwargs.get("effects", [])
        assert len(effects) == 1

    def test_multiple_hooks_all_registered(self):
        real_skill = Skill(
            name="alpha",
            description="d",
            body="Body",
            base_dir=None,
            hooks={
                "on_tool_error": "A",
                "query_end": "B",
                "before_llm_call": "C",
            },
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        parent = MagicMock()
        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        _invoke_execute(skill_tool, name="alpha", args="")
        from obelix.core.agent.hooks import AgentEvent

        expected_events = {
            AgentEvent.ON_TOOL_ERROR,
            AgentEvent.QUERY_END,
            AgentEvent.BEFORE_LLM_CALL,
        }
        called_events = {c.args[0] for c in parent.on.call_args_list}
        assert expected_events <= called_events

    def test_hooks_not_re_registered_on_second_invocation(self):
        """Idempotence also covers hook registration — not duplicated."""
        real_skill = Skill(
            name="alpha",
            description="d",
            body="Body",
            base_dir=None,
            hooks={"on_tool_error": "Retry"},
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        parent = MagicMock()
        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        _invoke_execute(skill_tool, name="alpha", args="")
        n_calls_first = parent.on.call_count
        _invoke_execute(skill_tool, name="alpha", args="")
        # Second invocation must NOT re-register
        assert parent.on.call_count == n_calls_first

    def test_no_parent_agent_hooks_silently_skipped(self):
        """When parent_agent=None, hooks are silently not registered (log-only)."""
        real_skill = Skill(
            name="alpha",
            description="d",
            body="Body",
            base_dir=None,
            hooks={"on_tool_error": "Retry"},
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        skill_tool = make_skill_tool(mgr, parent_agent=None)  # no parent
        r = _invoke_execute(skill_tool, name="alpha", args="")
        # Execution must succeed regardless
        from obelix.core.model.tool_message import ToolStatus

        assert r.status == ToolStatus.SUCCESS
        assert r.result == "Body"


class TestHookRegistrationRealAgent:
    """Integration: register against a real BaseAgent.on() + Hook.handle()."""

    def test_real_agent_receives_hook_and_executes_effect(self):
        """Register a hook via skill, fire the event, verify effect injects HumanMessage."""
        import asyncio

        from obelix.core.agent.base_agent import BaseAgent
        from obelix.core.agent.hooks import AgentEvent, AgentStatus

        # Build a minimal BaseAgent without running its heavy __init__ —
        # only the hooks registry + conversation_history are exercised here.
        parent = BaseAgent.__new__(BaseAgent)
        parent._hooks = {event: [] for event in AgentEvent}
        parent.conversation_history = []

        real_skill = Skill(
            name="alpha",
            description="d",
            body="Body",
            base_dir=None,
            hooks={"on_tool_error": "Retry with care."},
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])

        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        _invoke_execute(skill_tool, name="alpha", args="")

        # Exactly one hook registered on ON_TOOL_ERROR
        hooks_on_error = parent._hooks[AgentEvent.ON_TOOL_ERROR]
        assert len(hooks_on_error) == 1

        # Fire the hook — its effect should append HumanMessage to history
        status = AgentStatus(event=AgentEvent.ON_TOOL_ERROR, agent=parent)
        asyncio.run(hooks_on_error[0].execute(status))

        # conversation_history received the injected HumanMessage
        assert len(parent.conversation_history) == 1
        msg = parent.conversation_history[0]
        assert msg.content == "Retry with care."


class TestHookCleanupOnQueryEnd:
    """Skill-scoped hooks are removed at QUERY_END so subsequent queries start clean."""

    def test_cleanup_hook_registered_once_per_tool_instance(self):
        """QUERY_END cleanup hook is installed exactly once, not per skill invocation."""
        real_skill_a = Skill(
            name="a",
            description="d",
            body="A",
            base_dir=None,
            hooks={"on_tool_error": "retry A"},
        )
        real_skill_b = Skill(
            name="b",
            description="d",
            body="B",
            base_dir=None,
            hooks={"on_tool_error": "retry B"},
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill_a, real_skill_b]
        mgr = SkillManager(providers=[provider])
        parent = MagicMock()
        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        _invoke_execute(skill_tool, name="a", args="")
        _invoke_execute(skill_tool, name="b", args="")
        # Count parent.on(AgentEvent.QUERY_END) calls
        from obelix.core.agent.hooks import AgentEvent

        query_end_calls = [
            c for c in parent.on.call_args_list if c.args[0] == AgentEvent.QUERY_END
        ]
        assert len(query_end_calls) == 1

    def test_cleanup_not_registered_when_no_skill_has_hooks(self):
        """If no skill has hooks, no cleanup hook is needed."""
        real_skill = Skill(
            name="plain",
            description="d",
            body="Body",
            base_dir=None,
            hooks={},
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        parent = MagicMock()
        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        _invoke_execute(skill_tool, name="plain", args="")
        from obelix.core.agent.hooks import AgentEvent

        query_end_calls = [
            c for c in parent.on.call_args_list if c.args[0] == AgentEvent.QUERY_END
        ]
        assert len(query_end_calls) == 0

    def test_cleanup_fires_unregister_for_each_skill_hook_real_agent(self):
        """Integration: QUERY_END cleanup actually removes the registered hooks."""
        import asyncio

        from obelix.core.agent.base_agent import BaseAgent
        from obelix.core.agent.hooks import AgentEvent, AgentStatus

        parent = BaseAgent.__new__(BaseAgent)
        parent._hooks = {event: [] for event in AgentEvent}
        parent.conversation_history = []

        real_skill = Skill(
            name="a",
            description="d",
            body="A",
            base_dir=None,
            hooks={"on_tool_error": "retry", "before_llm_call": "pre"},
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        _invoke_execute(skill_tool, name="a", args="")

        # Before cleanup: 2 skill hooks + 1 query_end cleanup hook registered
        assert len(parent._hooks[AgentEvent.ON_TOOL_ERROR]) == 1
        assert len(parent._hooks[AgentEvent.BEFORE_LLM_CALL]) == 1
        assert len(parent._hooks[AgentEvent.QUERY_END]) == 1

        # Fire QUERY_END — cleanup should remove all skill-scoped hooks
        cleanup_hook = parent._hooks[AgentEvent.QUERY_END][0]
        status = AgentStatus(event=AgentEvent.QUERY_END, agent=parent)
        asyncio.run(cleanup_hook.execute(status))

        # After cleanup: skill hooks removed, QUERY_END also self-removed
        assert len(parent._hooks[AgentEvent.ON_TOOL_ERROR]) == 0
        assert len(parent._hooks[AgentEvent.BEFORE_LLM_CALL]) == 0
        assert len(parent._hooks[AgentEvent.QUERY_END]) == 0

    def test_cleanup_resets_active_skills_for_next_query(self):
        """After QUERY_END fires, invoking the same skill again returns the body (not 'already active')."""
        import asyncio

        from obelix.core.agent.base_agent import BaseAgent
        from obelix.core.agent.hooks import AgentEvent, AgentStatus

        parent = BaseAgent.__new__(BaseAgent)
        parent._hooks = {event: [] for event in AgentEvent}
        parent.conversation_history = []

        real_skill = Skill(
            name="a",
            description="d",
            body="A body",
            base_dir=None,
            hooks={"on_tool_error": "retry"},
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        r1 = _invoke_execute(skill_tool, name="a", args="")
        assert r1.result == "A body"
        r2 = _invoke_execute(skill_tool, name="a", args="")
        assert "already active" in r2.result.lower()

        # Fire QUERY_END cleanup
        cleanup_hook = parent._hooks[AgentEvent.QUERY_END][0]
        status = AgentStatus(event=AgentEvent.QUERY_END, agent=parent)
        asyncio.run(cleanup_hook.execute(status))

        # Now a fresh invocation should succeed normally (not "already active")
        r3 = _invoke_execute(skill_tool, name="a", args="")
        assert r3.result == "A body"

    def test_two_query_cycles_each_installs_fresh_cleanup_hook(self):
        """After Query 1 cleanup fires, Query 2 invocation installs a NEW cleanup hook."""
        import asyncio

        from obelix.core.agent.base_agent import BaseAgent
        from obelix.core.agent.hooks import AgentEvent, AgentStatus

        parent = BaseAgent.__new__(BaseAgent)
        parent._hooks = {event: [] for event in AgentEvent}
        parent.conversation_history = []

        real_skill = Skill(
            name="a",
            description="d",
            body="A",
            base_dir=None,
            hooks={"on_tool_error": "retry"},
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        skill_tool = make_skill_tool(mgr, parent_agent=parent)

        # Query 1
        _invoke_execute(skill_tool, name="a", args="")
        assert len(parent._hooks[AgentEvent.ON_TOOL_ERROR]) == 1
        assert len(parent._hooks[AgentEvent.QUERY_END]) == 1
        cleanup1 = parent._hooks[AgentEvent.QUERY_END][0]
        asyncio.run(
            cleanup1.execute(AgentStatus(event=AgentEvent.QUERY_END, agent=parent))
        )
        # All gone after cleanup
        assert len(parent._hooks[AgentEvent.ON_TOOL_ERROR]) == 0
        assert len(parent._hooks[AgentEvent.QUERY_END]) == 0

        # Query 2 — invoking the skill again must install a FRESH cleanup hook
        _invoke_execute(skill_tool, name="a", args="")
        assert len(parent._hooks[AgentEvent.ON_TOOL_ERROR]) == 1
        assert len(parent._hooks[AgentEvent.QUERY_END]) == 1
        cleanup2 = parent._hooks[AgentEvent.QUERY_END][0]
        # Must be a new hook instance, not the previous one
        assert cleanup2 is not cleanup1


class TestForkExecution:
    """Skills with context='fork' run in an isolated sub-agent."""

    def _fork_skill(self, body: str = "Fork body", hooks: dict | None = None) -> Skill:
        return Skill(
            name="fork_skill",
            description="A forked skill",
            body=body,
            base_dir=None,
            context="fork",
            hooks=hooks or {},
        )

    def test_fork_skill_invokes_subagent(self, monkeypatch):
        """When context=='fork', SubAgentWrapper is used and the result flows back."""
        from obelix.plugins.builtin import skill_tool as st_mod

        real_skill = self._fork_skill(body="Execute the protocol")
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])

        parent = MagicMock()
        parent.provider = MagicMock()
        parent.registered_tools = []
        parent.max_iterations = 15

        # Track what SubAgentWrapper receives + pretend it succeeded
        calls = {}

        class _FakeSub:
            def __init__(self, agent, *, name, description, stateless=False):
                calls["agent_system_message"] = str(agent.system_message.content)
                calls["name"] = name
                calls["description"] = description
                calls["stateless"] = stateless

            async def execute(self, tool_call):
                from obelix.core.model.tool_message import ToolResult, ToolStatus

                calls["tool_call"] = tool_call
                return ToolResult(
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    result="Fork result text",
                    status=ToolStatus.SUCCESS,
                )

        monkeypatch.setattr(st_mod, "SubAgentWrapper", _FakeSub)

        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        result = _invoke_execute(skill_tool, name="fork_skill", args="")

        from obelix.core.model.tool_message import ToolStatus

        assert result.status == ToolStatus.SUCCESS
        assert result.result == "Fork result text"
        assert calls["stateless"] is True
        assert "Execute the protocol" in calls["agent_system_message"]
        assert calls["name"] == "fork_skill"

    def test_fork_skill_inherits_provider_and_tools(self, monkeypatch):
        """Inner agent sees the parent's provider and registered_tools."""
        from obelix.plugins.builtin import skill_tool as st_mod

        real_skill = self._fork_skill()
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])

        parent_provider = MagicMock()
        parent_tool = MagicMock()
        parent_tool.tool_name = "parent_tool"
        parent = MagicMock()
        parent.provider = parent_provider
        parent.registered_tools = [parent_tool]
        parent.max_iterations = 7

        captured = {}

        class _FakeSub:
            def __init__(self, agent, *, name, description, stateless=False):
                captured["provider"] = agent.provider
                captured["tool_names"] = [
                    getattr(t, "tool_name", None) for t in agent.registered_tools
                ]
                captured["max_iter"] = agent.max_iterations

            async def execute(self, tool_call):
                from obelix.core.model.tool_message import ToolResult, ToolStatus

                return ToolResult(
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    result="ok",
                    status=ToolStatus.SUCCESS,
                )

        monkeypatch.setattr(st_mod, "SubAgentWrapper", _FakeSub)

        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        _invoke_execute(skill_tool, name="fork_skill", args="")

        assert captured["provider"] is parent_provider
        assert "parent_tool" in captured["tool_names"]
        assert captured["max_iter"] == 7

    def test_fork_skill_args_and_placeholders_substituted(self, monkeypatch):
        """$ARGUMENTS and meta placeholders are resolved BEFORE the body hits the sub-agent."""
        from obelix.plugins.builtin import skill_tool as st_mod

        real_skill = Skill(
            name="fork_skill",
            description="d",
            body="args=$ARGUMENTS sid=${OBELIX_SESSION_ID}",
            base_dir=None,
            context="fork",
        )
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])

        parent = MagicMock()
        parent.provider = MagicMock()
        parent.registered_tools = []
        parent.max_iterations = 15

        captured = {}

        class _FakeSub:
            def __init__(self, agent, *, name, description, stateless=False):
                captured["system_message"] = str(agent.system_message.content)

            async def execute(self, tool_call):
                from obelix.core.model.tool_message import ToolResult, ToolStatus

                return ToolResult(
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    result="ok",
                    status=ToolStatus.SUCCESS,
                )

        monkeypatch.setattr(st_mod, "SubAgentWrapper", _FakeSub)

        skill_tool = make_skill_tool(mgr, session_id="SID-X", parent_agent=parent)
        _invoke_execute(skill_tool, name="fork_skill", args="hello world")

        assert "args=hello world" in captured["system_message"]
        assert "sid=SID-X" in captured["system_message"]

    def test_fork_skill_returns_last_assistant_content(self, monkeypatch):
        """The sub-agent's ToolResult.result becomes the outer call's result."""
        from obelix.plugins.builtin import skill_tool as st_mod

        real_skill = self._fork_skill()
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        parent = MagicMock()
        parent.provider = MagicMock()
        parent.registered_tools = []
        parent.max_iterations = 15

        class _FakeSub:
            def __init__(self, *a, **kw):
                pass

            async def execute(self, tool_call):
                from obelix.core.model.tool_message import ToolResult, ToolStatus

                return ToolResult(
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    result="Deep investigation complete. Root cause: X.",
                    status=ToolStatus.SUCCESS,
                )

        monkeypatch.setattr(st_mod, "SubAgentWrapper", _FakeSub)

        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        result = _invoke_execute(skill_tool, name="fork_skill", args="")
        assert "Root cause: X" in result.result

    def test_fork_skill_idempotence(self, monkeypatch):
        """Second invocation of same fork skill returns 'already active'."""
        from obelix.plugins.builtin import skill_tool as st_mod

        real_skill = self._fork_skill()
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        parent = MagicMock()
        parent.provider = MagicMock()
        parent.registered_tools = []
        parent.max_iterations = 15

        class _FakeSub:
            def __init__(self, *a, **kw):
                pass

            async def execute(self, tool_call):
                from obelix.core.model.tool_message import ToolResult, ToolStatus

                return ToolResult(
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    result="first",
                    status=ToolStatus.SUCCESS,
                )

        monkeypatch.setattr(st_mod, "SubAgentWrapper", _FakeSub)

        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        r1 = _invoke_execute(skill_tool, name="fork_skill", args="")
        assert r1.result == "first"
        r2 = _invoke_execute(skill_tool, name="fork_skill", args="")
        assert "already active" in r2.result.lower()

    def test_fork_skill_without_parent_agent_returns_error(self):
        """Fork requires a parent context — no parent means ERROR status."""
        real_skill = self._fork_skill()
        provider = MagicMock()
        provider.discover.return_value = [real_skill]
        mgr = SkillManager(providers=[provider])
        skill_tool = make_skill_tool(mgr, parent_agent=None)
        result = _invoke_execute(skill_tool, name="fork_skill", args="")
        from obelix.core.model.tool_message import ToolStatus

        assert result.status == ToolStatus.ERROR


class TestForkExecutionRealAgent:
    """End-to-end fork exercising the real BaseAgent(...) construction path."""

    def test_fork_with_real_base_agent_and_stub_provider(self):
        """Full path: real BaseAgent + real SubAgentWrapper + stub LLM provider."""
        from obelix.core.agent.base_agent import BaseAgent
        from obelix.core.model.assistant_message import AssistantMessage
        from obelix.infrastructure.providers import Providers
        from obelix.ports.outbound.llm_provider import AbstractLLMProvider

        class _StubProvider(AbstractLLMProvider):
            """Minimal provider: returns a single AssistantMessage and stops."""

            model_id = "stub-model-1"

            @property
            def provider_type(self):
                return Providers.ANTHROPIC

            async def invoke(self, messages, tools, response_schema=None):
                # Immediately return a final response — no tool calls.
                return AssistantMessage(content="Fork completed successfully.")

        parent = BaseAgent(
            system_message="You are the parent.",
            provider=_StubProvider(),
            max_iterations=3,
        )

        fork_skill = Skill(
            name="deep_analyzer",
            description="Runs a deep analysis in a sub-agent",
            body="You are a deep analyzer. Summarize X.",
            base_dir=None,
            context="fork",
        )
        provider = MagicMock()
        provider.discover.return_value = [fork_skill]
        mgr = SkillManager(providers=[provider])

        skill_tool = make_skill_tool(mgr, parent_agent=parent)
        result = _invoke_execute(skill_tool, name="deep_analyzer", args="")

        from obelix.core.model.tool_message import ToolStatus

        assert result.status == ToolStatus.SUCCESS
        assert result.result == "Fork completed successfully."
