from typing import List, Optional, Union, Type

from obelix.core.agent import BaseAgent
from obelix.core.tool.tool_base import ToolBase
from obelix.ports.outbound.llm_provider import AbstractLLMProvider


class CoordinatorAgent(BaseAgent):
    """Coordinator agent."""

    def __init__(
        self,
        system_message: str,
        provider: Optional[AbstractLLMProvider] = None,
        tools: Optional[Union[ToolBase, Type[ToolBase], List[Union[Type[ToolBase], ToolBase]]]] = None,
    ) -> None:
        super().__init__(
            system_message=system_message,
            provider=provider,
            tools=tools,
        )
