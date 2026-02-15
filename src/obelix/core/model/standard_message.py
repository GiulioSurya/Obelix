from typing import Union

from obelix.core.model.human_message import HumanMessage
from obelix.core.model.system_message import SystemMessage
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.tool_message import ToolMessage

# Type alias for all types of standardized messages
StandardMessage = Union[HumanMessage, SystemMessage, AssistantMessage, ToolMessage]