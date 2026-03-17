"""A2A-specific deferred tool for requesting input from the remote client.

When served via A2A (a2a_serve), this tool is automatically registered
on the agent. When the LLM invokes it, the tool returns None — which
signals the BaseAgent loop to stop and yield a StreamEvent with
deferred_tool_calls. The A2A executor then emits `input-required`.

This is the A2A counterpart of AskUserQuestionTool (which blocks on
stdin for CLI usage). Both share the same UX pattern: present a
question with structured options, wait for the answer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from obelix.core.tool.tool_decorator import tool


class QuestionOption(BaseModel):
    """A single option the client can choose from."""

    label: str = Field(
        ..., description="Short display text for this option (1-5 words)"
    )
    description: str = Field(..., description="Explanation of what this option means")


@tool(
    name="request_user_input",
    description=(
        "Request additional input from the user/client to continue the task. "
        "Use this when you need clarification, a choice between options, or "
        "any information the user hasn't provided yet. "
        "Present a clear question and, when applicable, structured options."
    ),
    is_deferred=True,
)
class RequestUserInputTool:
    """Deferred A2A tool that stops the agent loop until the client responds.

    The LLM provides a question and optional structured choices.
    Because is_deferred=True and execute() returns None, the BaseAgent
    loop stops and yields the tool call info. The A2A executor emits
    TaskState.input_required with the question text. When the client
    responds, the executor injects the answer as a ToolMessage and
    restarts the agent.
    """

    question: str = Field(
        ...,
        description=(
            "The question to ask. Should be clear, specific, and end with '?'. "
            "Example: 'Which currency do you want to convert to?'"
        ),
    )
    options: list[QuestionOption] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "Optional structured choices for the client. "
            "When provided, the client can pick from these or give a free-form answer. "
            "Omit when the question requires free-form input."
        ),
    )

    def execute(self) -> None:
        """Return None to signal deferred execution.

        The BaseAgent detects is_deferred=True + None result and stops
        the loop, yielding the tool call for the caller to handle.
        """
        return None
