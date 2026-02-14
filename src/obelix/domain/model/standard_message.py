from typing import Union

from obelix.domain.model.human_message import HumanMessage
from obelix.domain.model.system_message import SystemMessage
from obelix.domain.model.assistant_message import AssistantMessage
from obelix.domain.model.tool_message import ToolMessage

# Type alias for all types of standardized messages
StandardMessage = Union[HumanMessage, SystemMessage, AssistantMessage, ToolMessage]