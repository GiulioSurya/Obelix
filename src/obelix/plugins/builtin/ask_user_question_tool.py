# src/tools/tool/ask_user_question_tool.py
"""
Tool for asking interactive questions to the user.

Allows the agent to gather information, clarify requirements, and get decisions
from the user through structured questions with multiple choice options.
"""
import asyncio
from typing import List

from pydantic import BaseModel, Field

from obelix.core.tool.tool_decorator import tool
from obelix.core.tool.tool_base import ToolBase


class QuestionOption(BaseModel):
    """Single option for a question."""
    label: str = Field(
        ...,
        description="Display text for this option (1-5 words, concise)"
    )
    description: str = Field(
        ...,
        description="Explanation of what this option means or what happens if chosen"
    )



class Question(BaseModel):
    """A question to ask the user."""
    question: str = Field(
        ...,
        description="The complete question to ask. Should be clear and end with '?'"
    )
    header: str = Field(
        ...,
        max_length=12,
        description="Short label displayed as chip/tag (max 12 chars). E.g.: 'Auth method', 'Library'"
    )
    options: List[QuestionOption] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="2-4 available choices. Each must be distinct."
    )
    multi_select: bool = Field(
        default=False,
        description="If true, user can select multiple options. If false, only one."
    )


@tool(
    name="ask_user_question",
    description="""\
Ask interactive questions to the user and receive structured responses.
Use to clarify requirements, gather preferences, validate assumptions, or get decisions.

 l,"""
)
class AskUserQuestionTool(ToolBase):
    """
    Tool for asking interactive questions to the user.

    Presents questions with options and collects structured responses.
    Supports single-select and multi-select questions.
    User can always choose 'Other' for custom input.

    Example usage by LLM:
        questions=[
            Question(
                question="Which library should we use for date formatting?",
                header="Library",
                options=[
                    QuestionOption(label="date-fns", description="Lightweight, tree-shakeable"),
                    QuestionOption(label="moment.js", description="Full-featured, larger size"),
                    QuestionOption(label="dayjs", description="Moment-compatible, small")
                ],
                multi_select=False
            )
        ]
    """

    questions: List[Question] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="Questions to ask the user (1-4 questions)"
    )

    async def execute(self) -> dict:
        """
        Present questions to user and collect responses.

        Returns:
            dict with 'answers' mapping question text to user's answer(s)
        """
        answers = {}

        for question in self.questions:
            answer = await self._ask_single_question(question)
            answers[question.question] = answer

        return {"answers": answers}

    async def _ask_single_question(self, question: Question) -> str | List[str]:
        """
        Present a single question and get user response.

        Uses asyncio.to_thread to avoid blocking the event loop.

        Args:
            question: The Question object to present

        Returns:
            Selected label(s) or custom input string
        """
        return await asyncio.to_thread(self._blocking_ask, question)

    def _blocking_ask(self, question: Question) -> str | List[str]:
        """
        Blocking implementation of question asking.

        Args:
            question: The Question object to present

        Returns:
            Selected label(s) or custom input string
        """
        # Display header and question
        print(f"\n{'=' * 50}")
        print(f"[{question.header}] {question.question}")
        print("-" * 50)

        # Display numbered options with descriptions
        for i, opt in enumerate(question.options, 1):
            print(f"  {i}. {opt.label}")
            print(f"     └─ {opt.description}")

        # Add "Other" option
        other_num = len(question.options) + 1
        print(f"  {other_num}. Other (custom input)")

        # Show selection mode hint
        if question.multi_select:
            print(f"\n(Multi-select: enter comma-separated numbers, e.g. '1,3')")
        print()

        # Input loop
        while True:
            try:
                user_input = input("> ").strip()

                if not user_input:
                    print("Please enter a selection.")
                    continue

                if question.multi_select:
                    return self._parse_multi_select(user_input, question, other_num)
                else:
                    return self._parse_single_select(user_input, question, other_num)

            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")
            except KeyboardInterrupt:
                print("\nSelection cancelled.")
                return "Cancelled"

    def _parse_single_select(
        self,
        user_input: str,
        question: Question,
        other_num: int
    ) -> str:
        """Parse single selection input."""
        selection = int(user_input)

        if selection == other_num:
            return input("Enter your custom answer: ").strip()
        elif 1 <= selection <= len(question.options):
            return question.options[selection - 1].label
        else:
            raise ValueError(f"Enter a number between 1 and {other_num}")

    def _parse_multi_select(
        self,
        user_input: str,
        question: Question,
        other_num: int
    ) -> List[str]:
        """Parse multi-selection input."""
        selections = [int(x.strip()) for x in user_input.split(",")]
        selected_labels = []

        for sel in selections:
            if sel == other_num:
                custom = input("Enter your custom answer: ").strip()
                if custom:
                    selected_labels.append(custom)
            elif 1 <= sel <= len(question.options):
                label = question.options[sel - 1].label
                if label not in selected_labels:  # Avoid duplicates
                    selected_labels.append(label)
            else:
                raise ValueError(f"Invalid selection: {sel}")

        if not selected_labels:
            raise ValueError("No valid selections made")

        return selected_labels