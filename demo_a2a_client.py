"""A2A client test script using the a2a-sdk client library.

Tests the coordination protocol with a complex financial analysis
that exercises planning, multi-step calculation, and report generation.
"""

import asyncio
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


def _make_request(text: str, context_id: str | None = None) -> SendMessageRequest:
    """Build a SendMessageRequest with the given text."""
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


def _print_result(result, label: str = "") -> None:
    """Print task result in a readable format."""
    prefix = f"[{label}] " if label else ""
    if hasattr(result, "status"):
        task = result
        print(f"{prefix}Task ID: {task.id}")
        print(f"{prefix}Context ID: {task.context_id}")
        print(f"{prefix}State: {task.status.state}")

        if task.status.message:
            for part in task.status.message.parts:
                if hasattr(part.root, "text"):
                    print(f"{prefix}Agent says: {part.root.text[:800]}")

        if task.artifacts:
            for artifact in task.artifacts:
                for part in artifact.parts:
                    if hasattr(part.root, "text") and part.root.text:
                        text = part.root.text
                        print(
                            f"{prefix}Result: {text[:800]}..."
                            if len(text) > 800
                            else f"{prefix}Result: {text}"
                        )
        return task
    else:
        print(f"{prefix}Direct message: {result}")
        return result


async def _send_and_handle(client: A2AClient, query: str, label: str) -> None:
    """Send a message, handle input-required if needed, print result."""
    print(f"\n{'=' * 60}")
    print(f"[{label}] Sending: '{query[:120]}...'")
    print(f"{'=' * 60}")

    response = await client.send_message(_make_request(query))
    result = response.root.result

    task = _print_result(result, label)

    # Handle input-required flow
    if hasattr(task, "status") and task.status.state == TaskState.input_required:
        print(f"\n[{label}] Input required — sending confirmation...")
        answer = "Yes, proceed with all calculations and generate the full report"
        response2 = await client.send_message(
            _make_request(answer, context_id=task.context_id)
        )
        _print_result(response2.root.result, label)


async def main():
    base_url = "http://localhost:8001"

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as httpx_client:
        # Resolve agent card
        print("=== Resolving agent card ===")
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        card = await resolver.get_agent_card()
        print(f"Agent: {card.name}")
        print(f"Skills: {[s.name for s in card.skills]}")
        print()

        client = A2AClient(httpx_client=httpx_client, agent_card=card)

        # ── Complex financial analysis ──────────────────────────────────
        await _send_and_handle(
            client,
            (
                "I need a complete financial analysis for a company with 3 product lines. "
                "Product A: 1200 units sold at $45.99 each, cost $28.50 per unit. "
                "Product B: 850 units sold at $72.00 each, cost $41.25 per unit. "
                "Product C: 2100 units sold at $19.95 each, cost $12.80 per unit. "
                "For each product calculate: revenue, total cost, and profit. "
                "Then calculate overall company revenue, total cost, total profit, "
                "and profit margin percentage. "
                "Finally, generate a comprehensive financial report with all numbers, "
                "including which product is most profitable and which has the best margin."
            ),
            label="FINANCIAL",
        )

    print(f"\n{'=' * 60}")
    print("Test completed.")


if __name__ == "__main__":
    asyncio.run(main())
