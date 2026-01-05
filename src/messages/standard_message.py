from typing import Union

from src.messages.human_message import HumanMessage
from src.messages.system_message import SystemMessage
from src.messages.assistant_message import AssistantMessage
from src.messages.tool_message import ToolMessage

# Type alias for all types of standardized messages
StandardMessage = Union[HumanMessage, SystemMessage, AssistantMessage, ToolMessage]