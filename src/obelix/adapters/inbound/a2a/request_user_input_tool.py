"""A2A-specific tool for requesting input from the remote client.

When served via A2A (a2a_serve), this tool is automatically registered
on the agent. When the LLM invokes it, the tool suspends execution
until the A2A client sends a follow-up message on the same contextId.

This is the A2A counterpart of AskUserQuestionTool (which blocks on
stdin for CLI usage). Both share the same UX pattern: present a
question with structured options, wait for the answer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from obelix.adapters.inbound.a2a.input_channel import InputChannel, input_channel_var
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
)
class RequestUserInputTool:
    """A2A tool that suspends agent execution until the client responds.

    The LLM provides a question and optional structured choices.
    The tool suspends (via InputChannel) and the A2A executor emits
    an `input-required` task state. When the client sends a follow-up
    message, the tool resumes with the client's answer.
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

    async def execute(self) -> dict:
        """Suspend execution and wait for client input via the A2A channel.

        Returns:
            dict with 'answer' containing the client's response.

        Raises:
            RuntimeError: If not running inside an A2A context (no InputChannel).
            TimeoutError: If the client doesn't respond within the timeout.
        """
        channel: InputChannel | None = input_channel_var.get(None)
        if channel is None:
            raise RuntimeError(
                "request_user_input can only be used inside an A2A context. "
                "No InputChannel found in ContextVar."
            )

        # Build the full question text including options
        prompt = self._build_prompt()

        # Suspend until the client responds
        answer = await channel.request_input(prompt)

        return {"answer": answer}

    def _build_prompt(self) -> str:
        """Build the question text, appending options if present."""
        if not self.options:
            return self.question

        lines = [self.question, ""]
        for i, opt in enumerate(self.options, 1):
            lines.append(f"  {i}. {opt.label} — {opt.description}")
        lines.append("")
        lines.append(
            "You can pick one of the options above or provide a different answer."
        )
        return "\n".join(lines)
