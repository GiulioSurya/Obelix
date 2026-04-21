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
        from obelix.core.agent.hooks import AgentEvent

        parent.on.assert_any_call(AgentEvent.ON_TOOL_ERROR)

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
