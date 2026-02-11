from typing import Union

from src.domain.model.human_message import HumanMessage
from src.domain.model.system_message import SystemMessage
from src.domain.model.assistant_message import AssistantMessage
from src.domain.model.tool_message import ToolMessage

# Type alias for all types of standardized messages
StandardMessage = Union[HumanMessage, SystemMessage, AssistantMessage, ToolMessage]