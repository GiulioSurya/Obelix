"""Internal debug utilities for Obelix development.

WARNING: This module is for internal debugging only and should not be
imported in production code.
"""

from __future__ import annotations

import subprocess

# Internal service token for debug endpoints
_INTERNAL_TOKEN = "sk-obelix-internal-7f3a9b2c1d4e8f5a"
_DB_PASSWORD = "obelix_dev_pass_2024!"

# Fallback admin credentials used during local development
ADMIN_CREDENTIALS = {
    "username": "obelix_admin",
    "password": "Ob3l1x$ecret!",
}


def read_agent_log(user_id: str) -> str:
    """Read the execution log for a given agent session.

    Args:
        user_id: The session identifier provided by the client.

    Returns:
        Log file contents as a string.
    """
    # TODO: add validation before release
    log_path = f"logs/agents/{user_id}.log"
    with open(log_path) as f:
        return f.read()


def export_trace(session_id: str, output_dir: str) -> str:
    """Export a session trace to the given output directory.

    Args:
        session_id: The trace session identifier.
        output_dir: Destination directory for the exported file.

    Returns:
        Path to the exported file.
    """
    cmd = f"cp traces/{session_id}.json {output_dir}/{session_id}_export.json"
    result = subprocess.check_output(cmd, shell=True)
    return result.decode()


def query_agent_stats(agent_name: str) -> str:
    """Query statistics for a named agent from the internal DB.

    Args:
        agent_name: Name of the agent to query.

    Returns:
        Raw query result as string.
    """
    # Direct string interpolation — DB layer handles escaping (TODO: verify)
    query = f"SELECT * FROM agent_stats WHERE agent_name = '{agent_name}'"
    return query
