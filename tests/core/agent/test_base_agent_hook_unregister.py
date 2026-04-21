import pytest

from obelix.core.agent.base_agent import BaseAgent
from obelix.core.agent.hooks import AgentEvent, Hook


@pytest.fixture
def agent_stub():
    """Minimal BaseAgent for hook registry tests only — no provider init."""
    agent = BaseAgent.__new__(BaseAgent)
    agent._hooks = {event: [] for event in AgentEvent}
    return agent


class TestUnregisterHook:
    def test_remove_existing_hook(self, agent_stub):
        hook = Hook(AgentEvent.ON_TOOL_ERROR)
        agent_stub._hooks[AgentEvent.ON_TOOL_ERROR].append(hook)
        assert hook in agent_stub._hooks[AgentEvent.ON_TOOL_ERROR]
        agent_stub._unregister_hook(AgentEvent.ON_TOOL_ERROR, hook)
        assert hook not in agent_stub._hooks[AgentEvent.ON_TOOL_ERROR]

    def test_remove_missing_hook_no_error(self, agent_stub):
        hook = Hook(AgentEvent.ON_TOOL_ERROR)
        # Never registered — must not raise
        agent_stub._unregister_hook(AgentEvent.ON_TOOL_ERROR, hook)

    def test_remove_one_does_not_affect_others(self, agent_stub):
        h1 = Hook(AgentEvent.ON_TOOL_ERROR)
        h2 = Hook(AgentEvent.ON_TOOL_ERROR)
        agent_stub._hooks[AgentEvent.ON_TOOL_ERROR].extend([h1, h2])
        agent_stub._unregister_hook(AgentEvent.ON_TOOL_ERROR, h1)
        assert h1 not in agent_stub._hooks[AgentEvent.ON_TOOL_ERROR]
        assert h2 in agent_stub._hooks[AgentEvent.ON_TOOL_ERROR]

    def test_remove_by_identity_not_equality(self, agent_stub):
        """Two Hook instances with same event are distinct — remove only the exact one."""
        h1 = Hook(AgentEvent.ON_TOOL_ERROR)
        h2 = Hook(AgentEvent.ON_TOOL_ERROR)
        agent_stub._hooks[AgentEvent.ON_TOOL_ERROR].extend([h1, h2])
        agent_stub._unregister_hook(AgentEvent.ON_TOOL_ERROR, h2)
        assert h1 in agent_stub._hooks[AgentEvent.ON_TOOL_ERROR]
        assert h2 not in agent_stub._hooks[AgentEvent.ON_TOOL_ERROR]

    def test_wrong_event_key_noop(self, agent_stub):
        """Passing the wrong event key doesn't remove from other buckets."""
        h = Hook(AgentEvent.ON_TOOL_ERROR)
        agent_stub._hooks[AgentEvent.ON_TOOL_ERROR].append(h)
        agent_stub._unregister_hook(AgentEvent.QUERY_END, h)
        assert h in agent_stub._hooks[AgentEvent.ON_TOOL_ERROR]
