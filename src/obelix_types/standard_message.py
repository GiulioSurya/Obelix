from typing import Union

from src.obelix_types.human_message import HumanMessage
from src.obelix_types.system_message import SystemMessage
from src.obelix_types.assistant_message import AssistantMessage
from src.obelix_types.tool_message import ToolMessage

# Type alias for all types of standardized obelix_types
StandardMessage = Union[HumanMessage, SystemMessage, AssistantMessage, ToolMessage]