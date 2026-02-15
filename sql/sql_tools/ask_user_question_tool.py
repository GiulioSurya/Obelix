# src/tools/tool/ask_user_question_tool.py
"""
Tool per porre domande interattive all'utente.

Permette all'agente di raccogliere informazioni, chiarire requisiti e ottenere
decisioni dall'utente tramite domande strutturate con opzioni a scelta multipla.

Supporta due modalità operative:
- Modalità console: Usa input() per interazioni CLI (default)
- Modalità WebSocket: Invia domande al frontend quando session_id è impostato
"""
import asyncio
from typing import List

from pydantic import BaseModel, Field
from obelix.core.tool import ToolBase,tool

from sql.sql_tools.user_question_bridge import get_question_bridge
from sql.session_context import get_session_or_none


class Question(BaseModel):
    """Una domanda da porre all'utente."""
    question: str = Field(
        ...,
        description="La domanda completa da porre. Deve essere chiara e terminare con '?'"
    )
    options: List[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="scelte disponibili come stringhe semplici. Devono essere distinte."
    )
    multi_select: bool = Field(
        default=False,
        description="Se true, l'utente può selezionare più opzioni. Se false, solo una."
    )


@tool(
    name="ask_user_question",
    description=(
        "Usalo per chiarire requisiti, raccogliere preferenze, validare assunzioni "
        "o ottenere decisioni sulle scelte implementative. "
        "IMPORTANTE: NON includere 'Altro', 'Other' o opzioni simili di fallback - "
        "il sistema aggiunge automaticamente un pulsante 'Altro' per input personalizzati. "
        "Fornisci solo le opzioni specifiche e concrete pertinenti alla domanda."
    )
)
class AskUserQuestionTool(ToolBase):
    """
    Tool per porre domande interattive all'utente.

    Presenta domande con opzioni e raccoglie risposte strutturate.
    Supporta domande a selezione singola e multipla.
    L'utente può sempre scegliere 'Altro' per input personalizzato.

    Esempio JSON per LLM:
        {"questions": [{"question": "Quale DB?", "options": ["Oracle", "PostgreSQL", "MySQL"]}]}
    """

    questions: List[Question] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="Domande da porre all'utente (1-4 domande)"
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

        Automatically detects context:
        - If session is active with WebSocket: Uses WebSocket to send to frontend
        - Otherwise: Uses input() for console mode

        Args:
            question: The Question object to present

        Returns:
            Selected label(s) or custom input string
        """
        session = get_session_or_none()
        bridge = get_question_bridge()

        # Check if we're in WebSocket mode (session exists and has registered websocket)
        if session and bridge.get_websocket(session.session_id):
            # WebSocket mode: send to frontend
            return await self._ask_via_websocket(session.session_id, question)
        else:
            # Console mode: use blocking input
            return await asyncio.to_thread(self._blocking_ask, question)

    async def _ask_via_websocket(
        self,
        session_id: str,
        question: Question
    ) -> str | List[str]:
        """
        Send question to frontend via WebSocket and wait for response.

        Args:
            session_id: Current session identifier
            question: The Question object to present

        Returns:
            Selected label(s) or custom input string
        """
        bridge = get_question_bridge()
        return await bridge.send_question(
            session_id=session_id,
            question=question.question,
            options=question.options,
            multi_select=question.multi_select
        )

    def _blocking_ask(self, question: Question) -> str | List[str]:
        """
        Blocking implementation of question asking.

        Args:
            question: The Question object to present

        Returns:
            Selected label(s) or custom input string
        """
        # Display question
        print(f"\n{'=' * 50}")
        print(f"{question.question}")
        print("-" * 50)

        # Display numbered options
        for i, opt in enumerate(question.options, 1):
            print(f"  {i}. {opt}")

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
            return question.options[selection - 1]
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
        selected_options = []

        for sel in selections:
            if sel == other_num:
                custom = input("Enter your custom answer: ").strip()
                if custom:
                    selected_options.append(custom)
            elif 1 <= sel <= len(question.options):
                option = question.options[sel - 1]
                if option not in selected_options:  # Avoid duplicates
                    selected_options.append(option)
            else:
                raise ValueError(f"Invalid selection: {sel}")

        if not selected_options:
            raise ValueError("No valid selections made")

        return selected_options
