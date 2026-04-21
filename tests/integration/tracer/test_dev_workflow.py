"""End-to-end trace shape for the dev_workflow scenario (no real A2A HTTP).

Exercises the full span taxonomy produced by the CoordinatorAgent + 3
sub-agents + skills pipeline from ``examples/dev_workflow_server.py``,
with mocked LLM providers so the scenario replays deterministically.

Invariants asserted:

* exactly 3 ``sub_agent`` spans (one per registered sub-agent invocation);
* at least 4 ``agent`` spans (coordinator + 3 sub-agents at minimum, more
  when fork-mode skills open inner agent spans);
* at least 2 ``skill`` spans (code-review + commit-writer both fork,
  plus security-check inline = 3 in practice but we only require >= 2);
* exactly 1 ``human`` + 1 ``assistant`` span (root-only, not per sub-agent);
* no ``llm`` spans (LLM usage aggregated onto agent metadata since Task 10);
* ``memory.pull`` events on downstream agents (CommitAgent pulls from
  reviewer; SummaryAgent pulls from reviewer and commit_writer);
* ``memory.publish`` events on each agent that produced a final response.
"""

from __future__ import annotations

import pytest

from obelix.core.tracer.models import SpanType


@pytest.mark.asyncio
async def test_dev_workflow_span_taxonomy(dev_workflow_agents_with_spy):
    """The full pipeline produces the expected span types in the right counts."""
    coordinator, spy = dev_workflow_agents_with_spy
    await coordinator.execute_query_async("review my staged changes")

    agent_count = sum(1 for s in spy.spans if s.span_type == SpanType.agent)
    sub_agent_count = sum(1 for s in spy.spans if s.span_type == SpanType.sub_agent)
    skill_count = sum(1 for s in spy.spans if s.span_type == SpanType.skill)
    human_count = sum(1 for s in spy.spans if s.span_type == SpanType.human)
    assistant_count = sum(1 for s in spy.spans if s.span_type == SpanType.assistant)
    tool_count = sum(1 for s in spy.spans if s.span_type == SpanType.tool)

    # Root agent (coordinator) + 3 sub-agents. Fork skills add inner agent
    # spans (one per fork = 2 more, for code-review and commit-writer) so the
    # total is 6 in the happy path. We only require >= 4 so future tweaks
    # that collapse fork inner agents don't break this test.
    assert agent_count >= 4, (
        f"expected >= 4 agent spans (coordinator + 3 sub-agents + fork inners), "
        f"got {agent_count}. Spans: "
        f"{[(s.span_type.value, s.name) for s in spy.spans]}"
    )

    # Exactly one SubAgentWrapper invocation per sub-agent.
    assert sub_agent_count == 3, (
        f"expected exactly 3 sub_agent spans (reviewer, commit_writer, summary), "
        f"got {sub_agent_count}"
    )

    # code-review (fork) + security-check (inline) + commit-writer (fork) = 3.
    # We require >= 2 to match the task 22 spec.
    assert skill_count >= 2, (
        f"expected >= 2 skill spans (code-review, commit-writer, security-check), "
        f"got {skill_count}"
    )

    # Root-only human/assistant spans.
    assert human_count == 1, (
        f"expected exactly 1 human span at the root, got {human_count}"
    )
    assert assistant_count == 1, (
        f"expected exactly 1 assistant span at the root, got {assistant_count}"
    )

    # No per-call LLM spans — usage is aggregated on agent span metadata.
    llm_count = sum(1 for s in spy.spans if str(s.span_type) == "llm")
    assert llm_count == 0, (
        f"expected 0 llm spans (usage aggregated onto agent.metadata.llm_usage), "
        f"got {llm_count}"
    )

    # Regular tool spans are emitted only for non-skill, non-sub_agent tools.
    # The fixture doesn't register any such tool, so the count should be 0.
    assert tool_count == 0, (
        f"expected 0 plain tool spans (all tool calls are skills or sub-agents), "
        f"got {tool_count}. Tool span names: "
        f"{[s.name for s in spy.spans if s.span_type == SpanType.tool]}"
    )


@pytest.mark.asyncio
async def test_dev_workflow_memory_events_on_downstream(
    dev_workflow_agents_with_spy,
):
    """Downstream agents emit ``memory.pull`` events, upstream emit ``memory.publish``.

    * CommitAgent has a predecessor (reviewer) → must pull from it.
    * SummaryAgent has two predecessors (reviewer + commit_writer) → must
      pull from at least reviewer (the scenario scripts reviewer to complete
      first and commit_writer to complete before summary).
    * Each agent producing a final response must emit at least one
      ``memory.publish`` event (kind=final).
    """
    coordinator, spy = dev_workflow_agents_with_spy
    await coordinator.execute_query_async("review my staged changes")

    # Group agent spans by their class name (the span.name carries the class
    # name at start-span time per start_agent_trace).
    agent_spans = [s for s in spy.spans if s.span_type == SpanType.agent]
    agents_by_name: dict[str, list] = {}
    for s in agent_spans:
        # span.name is agent_class_name for agent spans
        agents_by_name.setdefault(s.name, []).append(s)

    # --- CommitAgent: pulls from reviewer ---
    commit_spans = agents_by_name.get("_CommitAgent", [])
    assert commit_spans, (
        f"expected at least one agent span for _CommitAgent; "
        f"available: {sorted(agents_by_name.keys())}"
    )
    commit_pulls = [
        e for s in commit_spans for e in s.events if e.name == "memory.pull"
    ]
    assert any(e.attributes.get("from_agent") == "reviewer" for e in commit_pulls), (
        f"_CommitAgent did not pull from reviewer. pulls={commit_pulls}"
    )

    # --- SummaryAgent: pulls from reviewer (and ideally commit_writer) ---
    summary_spans = agents_by_name.get("_SummaryAgent", [])
    assert summary_spans, (
        f"expected at least one agent span for _SummaryAgent; "
        f"available: {sorted(agents_by_name.keys())}"
    )
    summary_pulls = [
        e for s in summary_spans for e in s.events if e.name == "memory.pull"
    ]
    summary_sources = {e.attributes.get("from_agent") for e in summary_pulls}
    assert "reviewer" in summary_sources, (
        f"_SummaryAgent did not pull from reviewer. sources={summary_sources}"
    )
    # commit_writer must also be present since the scenario ordering has
    # commit_writer completing before summary is invoked.
    assert "commit_writer" in summary_sources, (
        f"_SummaryAgent did not pull from commit_writer. sources={summary_sources}"
    )


@pytest.mark.asyncio
async def test_dev_workflow_memory_publish_events(dev_workflow_agents_with_spy):
    """Each agent that produced a final response emits ``memory.publish``.

    Only agents wired into the memory graph (reviewer, commit_writer, summary)
    emit publish events — the coordinator isn't in the graph so it should NOT.
    """
    coordinator, spy = dev_workflow_agents_with_spy
    await coordinator.execute_query_async("review my staged changes")

    agent_spans = [s for s in spy.spans if s.span_type == SpanType.agent]
    publish_events = [
        (s.name, e) for s in agent_spans for e in s.events if e.name == "memory.publish"
    ]

    agent_names_with_publish = {name for name, _ in publish_events}

    # The three wired sub-agents must each publish at least one final response.
    for required in ("_ReviewerAgent", "_CommitAgent", "_SummaryAgent"):
        assert required in agent_names_with_publish, (
            f"expected memory.publish event on {required}; "
            f"saw: {agent_names_with_publish}"
        )


@pytest.mark.asyncio
async def test_dev_workflow_skill_spans_carry_mode_metadata(
    dev_workflow_agents_with_spy,
):
    """Skill spans carry ``mode`` and ``source`` metadata from the skill frontmatter."""
    coordinator, spy = dev_workflow_agents_with_spy
    await coordinator.execute_query_async("review my staged changes")

    skill_spans = [s for s in spy.spans if s.span_type == SpanType.skill]
    assert skill_spans, "no skill spans emitted"

    modes = {s.name: s.metadata.get("mode") for s in skill_spans}
    sources = {s.name: s.metadata.get("source") for s in skill_spans}

    # Fork skills declare context=fork in their SKILL.md — metadata.mode must reflect it.
    assert modes.get("code-review") == "fork", (
        f"code-review mode mismatch; modes={modes}"
    )
    assert modes.get("commit-writer") == "fork", (
        f"commit-writer mode mismatch; modes={modes}"
    )
    # security-check is inline.
    if "security-check" in modes:
        assert modes["security-check"] == "inline", (
            f"security-check mode mismatch; modes={modes}"
        )

    # All filesystem-backed skills should report source=filesystem.
    for name, src in sources.items():
        assert src == "filesystem", (
            f"skill {name!r} has source={src!r}, expected 'filesystem'"
        )


@pytest.mark.asyncio
async def test_dev_workflow_sub_agent_spans_named_by_registered_name(
    dev_workflow_agents_with_spy,
):
    """``sub_agent`` spans are named after the registered tool name, not the class.

    The SubAgentWrapper attaches the sub-agent under the name given in
    ``register()`` — the sub_agent span should use that name so downstream
    trace consumers can correlate sub-agent invocations with the registry.
    """
    coordinator, spy = dev_workflow_agents_with_spy
    await coordinator.execute_query_async("review my staged changes")

    sub_agent_spans = [s for s in spy.spans if s.span_type == SpanType.sub_agent]
    names = sorted(s.name for s in sub_agent_spans)
    assert names == ["commit_writer", "reviewer", "summary"], (
        f"expected sub_agent spans for reviewer, commit_writer, summary; got {names}"
    )
