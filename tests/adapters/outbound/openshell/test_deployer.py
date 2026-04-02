"""Tests for OpenShellDeployer.

All tests mock the openshell SDK and CLI — no real Gateway or sandbox required.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


class TestDeploymentInfo:
    """DeploymentInfo is a frozen dataclass with sandbox_name, endpoint, port."""

    def test_construction(self):
        from obelix.adapters.outbound.openshell.deployer import DeploymentInfo

        info = DeploymentInfo(
            sandbox_name="test-sb",
            endpoint="http://localhost:8002",
            port=8002,
        )
        assert info.sandbox_name == "test-sb"
        assert info.endpoint == "http://localhost:8002"
        assert info.port == 8002

    def test_frozen(self):
        from obelix.adapters.outbound.openshell.deployer import DeploymentInfo

        info = DeploymentInfo(
            sandbox_name="sb", endpoint="http://localhost:8002", port=8002
        )
        with pytest.raises(FrozenInstanceError):
            info.sandbox_name = "other"

    def test_importable_from_package(self):
        from obelix.adapters.outbound.openshell import DeploymentInfo

        info = DeploymentInfo(
            sandbox_name="sb", endpoint="http://localhost:8002", port=8002
        )
        assert info.sandbox_name == "sb"
