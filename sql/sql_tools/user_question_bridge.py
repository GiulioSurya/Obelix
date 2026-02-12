# src/tools/user_question_bridge.py
"""
Bridge Singleton for WebSocket-Tool coordination.

Coordinates communication between WebSocket connections and AskUserQuestionTool,
allowing the tool to send questions to the frontend and receive responses.

Note: Session management is now handled by src/session_context.py.
      Use get_session().session_id to get the current session ID.
"""
import asyncio
import uuid
from typing import Dict, List, Optional

from fastapi import WebSocket


class UserQuestionBridge:
    """
    Singleton bridge that coordinates WebSocket-Tool communication.

    Manages WebSocket connections by session_id and handles pending questions
    using asyncio Futures for async/await support.

    Usage:
        bridge = UserQuestionBridge()
        bridge.register_websocket(session_id, websocket)
        answer = await bridge.send_question(session_id, question_data)
        bridge.deliver_answer(question_id, answer)
    """

    _instance: Optional['UserQuestionBridge'] = None

    def __new__(cls) -> 'UserQuestionBridge':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._websockets: Dict[str, WebSocket] = {}
        self._pending_questions: Dict[str, asyncio.Future] = {}
        self._initialized = True

    def register_websocket(self, session_id: str, websocket: WebSocket) -> None:
        """
        Register a WebSocket connection for a session.

        Args:
            session_id: Unique session identifier
            websocket: FastAPI WebSocket connection
        """
        self._websockets[session_id] = websocket

    def unregister_websocket(self, session_id: str) -> None:
        """
        Unregister a WebSocket connection when session ends.

        Args:
            session_id: Session identifier to unregister
        """
        self._websockets.pop(session_id, None)
        # Cancel any pending questions for this session
        for question_id in list(self._pending_questions.keys()):
            if question_id.startswith(session_id):
                future = self._pending_questions.pop(question_id, None)
                if future and not future.done():
                    future.cancel()

    def get_websocket(self, session_id: str) -> Optional[WebSocket]:
        """
        Get WebSocket connection for a session.

        Args:
            session_id: Session identifier

        Returns:
            WebSocket connection or None if not registered
        """
        return self._websockets.get(session_id)

    async def send_question(
        self,
        session_id: str,
        question: str,
        options: List[str],
        multi_select: bool = False
    ) -> str | List[str]:
        """
        Send a question to the frontend via WebSocket and wait for response.

        Creates a Future that will be resolved when deliver_answer is called.
        Adds "Other" option automatically (not included in LLM options).

        Args:
            session_id: Session identifier
            question: The question text
            options: List of option strings (without "Other")
            multi_select: Whether multiple selections are allowed

        Returns:
            User's answer (single string or list for multi-select)

        Raises:
            RuntimeError: If no WebSocket is registered for session
            asyncio.CancelledError: If the session is closed before answer
        """
        websocket = self._websockets.get(session_id)
        if websocket is None:
            raise RuntimeError(f"No WebSocket registered for session {session_id}")

        # Generate unique question ID
        question_id = f"{session_id}:{uuid.uuid4().hex[:8]}"

        # Create Future for the answer
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_questions[question_id] = future

        # Send question to frontend (add "Other" automatically)
        await websocket.send_json({
            "type": "question",
            "question_id": question_id,
            "question": question,
            "options": options,  # Frontend will add "Other" button
            "multi_select": multi_select
        })

        try:
            # Wait for answer (blocking until deliver_answer is called)
            answer = await future
            return answer
        finally:
            # Clean up
            self._pending_questions.pop(question_id, None)

    def deliver_answer(self, question_id: str, answer: str | List[str]) -> bool:
        """
        Deliver an answer to a pending question, resolving the Future.

        Args:
            question_id: The question ID from send_question
            answer: User's answer (string or list for multi-select)

        Returns:
            True if answer was delivered, False if question not found
        """
        future = self._pending_questions.get(question_id)
        if future is None:
            return False

        if not future.done():
            future.set_result(answer)
        return True

    def has_pending_question(self, question_id: str) -> bool:
        """
        Check if a question is pending.

        Args:
            question_id: Question identifier

        Returns:
            True if question is pending
        """
        return question_id in self._pending_questions


def get_question_bridge() -> UserQuestionBridge:
    """
    Get the singleton UserQuestionBridge instance.

    Returns:
        UserQuestionBridge singleton
    """
    return UserQuestionBridge()
