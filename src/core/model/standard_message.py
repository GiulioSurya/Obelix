from typing import Union

from src.core.model.human_message import HumanMessage
from src.core.model.system_message import SystemMessage
from src.core.model.assistant_message import AssistantMessage
from src.core.model.tool_message import ToolMessage

# Type alias for all types of standardized messages
StandardMessage = Union[HumanMessage, SystemMessage, AssistantMessage, ToolMessage]