try:
    from obelix.adapters.outbound.openshell.deployer import (
        DeploymentInfo,
        OpenShellDeployer,
    )
except ImportError:
    pass  # openshell extra not installed

__all__ = ["DeploymentInfo", "OpenShellDeployer"]
