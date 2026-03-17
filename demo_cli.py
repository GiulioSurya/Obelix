# demo_cli.py — Interactive CLI client for A2A agents
"""
Connect to one or more A2A agent servers and chat interactively.

Usage:
    uv run python demo_cli.py http://localhost:8002
    uv run python demo_cli.py http://localhost:8001 http://localhost:8002

Commands:
    /agents         List connected agents
    /switch <n>     Switch to agent n
    /quit           Exit
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
    TextPart,
)


class AgentConnection:
    """A resolved connection to a remote A2A agent."""

    def __init__(
        self,
        name: str,
        description: str,
        skills: list[str],
        client: A2AClient,
        url: str,
    ):
        self.name = name
        self.description = description
        self.skills = skills
        self.client = client
        self.url = url
        self.context_id: str | None = None  # persists across turns


def _make_request(text: str, context_id: str | None = None) -> SendMessageRequest:
    msg = Message(
        message_id=str(uuid.uuid4()),
        role=Role.user,
        parts=[Part(root=TextPart(text=text))],
        context_id=context_id,
    )
    return SendMessageRequest(
        id=str(uuid.uuid4()),
        params=MessageSendParams(message=msg),
    )


def _extract_text(task) -> str:
    """Extract text content from a task's artifacts."""
    if not hasattr(task, "artifacts") or not task.artifacts:
        return ""
    parts = []
    for artifact in task.artifacts:
        for part in artifact.parts:
            if hasattr(part.root, "text") and part.root.text:
                parts.append(part.root.text)
    return "".join(parts)


def _extract_status_message(task) -> str:
    """Extract text from task status message."""
    if not task.status.message:
        return ""
    parts = []
    for part in task.status.message.parts:
        if hasattr(part.root, "text"):
            parts.append(part.root.text)
    return "".join(parts)


async def resolve_agents(
    urls: list[str], httpx_client: httpx.AsyncClient
) -> list[AgentConnection]:
    """Resolve agent cards from URLs and create connections."""
    agents = []
    for url in urls:
        try:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
            card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=card)
            skills = [s.name for s in card.skills] if card.skills else []
            agents.append(
                AgentConnection(
                    name=card.name,
                    description=card.description or "",
                    skills=skills,
                    client=client,
                    url=url,
                )
            )
            print(f"  Connected: {card.name} ({url})")
        except Exception as e:
            print(f"  Failed: {url} — {e}")
    return agents


async def send_message(agent: AgentConnection, text: str) -> None:
    """Send a message and handle the response, including input-required."""
    request = _make_request(text, context_id=agent.context_id)
    response = await agent.client.send_message(request)
    task = response.root.result

    if not hasattr(task, "status"):
        print(f"[{agent.name}] {task}")
        return

    # Persist context for multi-turn
    agent.context_id = task.context_id

    # Handle input-required loop
    while task.status.state == TaskState.input_required:
        status_msg = _extract_status_message(task)

        # Try to parse as JSON (deferred tool protocol)
        display_msg = status_msg
        try:
            deferred_calls = json.loads(status_msg)
            if isinstance(deferred_calls, list):
                parts = []
                for call in deferred_calls:
                    tool_name = call.get("tool_name", "?")
                    args = call.get("arguments", {})
                    parts.append(f"[{tool_name}] {json.dumps(args, indent=2)}")
                display_msg = "\n".join(parts)
        except (json.JSONDecodeError, TypeError):
            pass

        print(f"\n[{agent.name}] Input required:")
        print(f"{display_msg}")
        print()

        answer = await asyncio.to_thread(input, "> ")
        if not answer.strip():
            answer = "proceed"

        request = _make_request(answer, context_id=agent.context_id)
        response = await agent.client.send_message(request)
        task = response.root.result

    # Print result
    if task.status.state == TaskState.completed:
        text = _extract_text(task)
        if text:
            print(f"\n[{agent.name}] {text}")
        else:
            print(f"\n[{agent.name}] (completed, no content)")
    elif task.status.state == TaskState.failed:
        msg = _extract_status_message(task)
        print(f"\n[{agent.name}] FAILED: {msg}")
    else:
        print(f"\n[{agent.name}] State: {task.status.state}")


def print_agents(agents: list[AgentConnection], current: int) -> None:
    print("\nAvailable agents:")
    for i, agent in enumerate(agents):
        marker = " *" if i == current else "  "
        skills_str = ", ".join(agent.skills[:3]) if agent.skills else "none"
        print(f"  [{i + 1}]{marker} {agent.name} — {agent.description[:60]}")
        print(f"        Skills: {skills_str}  |  {agent.url}")
    print()


async def main():
    urls = sys.argv[1:]
    if not urls:
        print("Usage: uv run python demo_cli.py <url1> [url2] [url3] ...")
        print("Example: uv run python demo_cli.py http://localhost:8002")
        sys.exit(1)

    print("Connecting to agents...")
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as httpx_client:
        agents = await resolve_agents(urls, httpx_client)

        if not agents:
            print("No agents connected. Exiting.")
            sys.exit(1)

        current = 0
        print_agents(agents, current)
        print(f"Active: {agents[current].name}")
        print("Commands: /agents, /switch <n>, /quit\n")

        while True:
            try:
                prompt = f"[{agents[current].name}] > "
                user_input = await asyncio.to_thread(input, prompt)
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            text = user_input.strip()
            if not text:
                continue

            if text == "/quit":
                print("Bye.")
                break

            if text == "/agents":
                print_agents(agents, current)
                continue

            if text.startswith("/switch"):
                parts = text.split()
                if len(parts) == 2 and parts[1].isdigit():
                    idx = int(parts[1]) - 1
                    if 0 <= idx < len(agents):
                        current = idx
                        print(f"Switched to: {agents[current].name}\n")
                    else:
                        print(f"Invalid. Use 1-{len(agents)}.")
                else:
                    print("Usage: /switch <n>")
                continue

            try:
                await send_message(agents[current], text)
            except Exception as e:
                print(f"Error: {e}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
