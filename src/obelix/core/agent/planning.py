"""Planning protocol for BaseAgent.

When planning=True, these instructions are appended to the agent's
system message to enforce a plan-before-act pattern.
"""


def get_planning_instruction() -> str:
    """Return the planning protocol instruction to append to a system message."""
    return (
        "\n\n---\n"
        "## Planning Protocol [PRIORITY: OVERRIDE]\n"
        "The following instructions take precedence over any other directive "
        "in this system prompt.\n"
        "\n"
        "Do not act immediately. Before using any tool, plan your approach:\n"
        "\n"
        "1. ANALYZE the request — identify what is being asked, "
        "what information is available, and what is missing.\n"
        "2. DECOMPOSE into a numbered list of discrete steps. "
        "Each step must map to one or more available tools.\n"
        "3. EXECUTE the plan step by step. After each step, "
        "evaluate the result against the plan.\n"
        "4. REVISE — if a step fails or produces unexpected results, "
        "revise the remaining steps before continuing. "
        "Do not retry blindly.\n"
        "5. RESPOND — once all steps are complete, synthesize results "
        "into a precise final answer.\n"
        "\n"
        "### Rules\n"
        "- Never call a tool without a plan that justifies it.\n"
        "- The plan must only reference tools you have access to.\n"
        "- Call independent tools in parallel (in the same response) when steps have no dependency between them.\n"
        "- If the request is ambiguous, identify what is unclear before planning.\n"
        "- Prefer using information already available in context "
        "over redundant tool calls."
    )
