"""OpenShell Deployer — deploys Obelix agents inside OpenShell sandboxes as A2A servers.

Uses the ``openshell`` Python SDK for sandbox CRUD and exec, and the ``openshell``
CLI for provider management, policy, and port forwarding.

Requires: ``uv sync --extra serve --extra openshell`` (Linux/macOS only).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentInfo:
    """Immutable result of a successful deployment."""

    sandbox_name: str
    endpoint: str  # "http://localhost:{port}"
    port: int
