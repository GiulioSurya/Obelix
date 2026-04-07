# examples/deploy_demo/deploy.py
"""Deploy a sandboxed BashTool agent inside an OpenShell sandbox.

This script uses a2a_openshell_deploy() to:
1. Register LLM provider credentials in the OpenShell gateway
2. Build a container image with the project code
3. Create a sandbox with the security policy
4. Start the A2A server inside the sandbox (examples.deploy_demo.serve)
5. Forward port 8002 to localhost

The client connects the same way as a2a_serve():
    uv run python examples/cli_client.py http://localhost:8002

Requirements (Linux/macOS or WSL2):
    - Docker running
    - OpenShell gateway started: openshell gateway start
    - Dependencies: uv sync --extra litellm --extra serve --extra openshell

Env vars (loaded from examples/shell-demo/.env):
    - ANTHROPIC_API_KEY: Anthropic API key
    - LITELLM_MODEL: model to use (default: anthropic/claude-haiku-4-5-20251001)

Usage:
    uv run python examples/deploy_demo/deploy.py
"""

from pathlib import Path

from dotenv import load_dotenv

from obelix.core.agent.agent_factory import AgentFactory
from obelix.infrastructure.logging import setup_logging

# Load env from shell-demo/.env (ANTHROPIC_API_KEY, LITELLM_MODEL, etc.)
load_dotenv(Path(__file__).parent.parent / "shell-demo" / ".env")
setup_logging(console_level="INFO")

if __name__ == "__main__":
    factory = AgentFactory()

    # No need to register agents or define classes here.
    # The entrypoint module (serve.py) handles all of that inside the sandbox.
    factory.a2a_openshell_deploy(
        "sandbox_bash",
        port=8002,
        entrypoint="examples.deploy_demo.serve",
        policy="examples/deploy_demo/policy.yaml",
        providers=["anthropic"],
        extras=["litellm"],
        description="Agent with sandboxed shell execution via OpenShell Deployer",
    )
