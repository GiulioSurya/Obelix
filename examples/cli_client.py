# examples/cli_client.py -- Interactive CLI client for A2A agents
"""
Connect to one or more A2A agent servers and chat interactively.

Usage:
    uv run python examples/cli_client.py http://localhost:8002
    uv run python examples/cli_client.py http://localhost:8001 http://localhost:8002
    uv run python examples/cli_client.py http://localhost:8002 --webhook-host host.docker.internal
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from obelix.adapters.inbound.a2a.client import CLIClient  # noqa: E402

if __name__ == "__main__":
    CLIClient.from_cli().run()
