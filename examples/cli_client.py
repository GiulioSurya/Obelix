# examples/cli_client.py -- Interactive CLI client for A2A agents
"""
Connect to one or more A2A agent servers and chat interactively.

Usage:
    uv run python examples/cli_client.py http://localhost:8002
    uv run python examples/cli_client.py http://localhost:8001 http://localhost:8002

Commands:
    /agents         List connected agents
    /switch <n>     Switch to agent n
    /clear          Clear conversation context
    /quit           Exit
"""

from __future__ import annotations

import sys

from obelix.adapters.inbound.a2a.client import CLIClient
from obelix.adapters.inbound.a2a.client.handlers import default_dispatcher

if __name__ == "__main__":
    app = CLIClient(dispatcher=default_dispatcher(), urls=sys.argv[1:])
    app.run()
