# examples/dev_workflow_server.py -- A2A server for dev workflow with skills
"""
A2A server with a dev workflow pipeline: code review → commit message → report.

Flow:
  reviewer_agent  -->  commit_agent  -->  summary_agent
       |                   |                   |
       +-------------------+-------------------+
                    coordinator (orchestrator)

ReviewerAgent  — reads staged diff via git_diff tool, runs two skills:
                 1. 'code-review'   (fork)   deep analysis in isolated sub-agent
                 2. 'security-check' (inline) quick vulnerability scan
CommitAgent    — takes review findings from shared memory, produces a
                 conventional commit message via 'commit-writer' skill (fork)
SummaryAgent   — formats review + commit into a developer-ready markdown report

Requirements:
    uv sync --extra litellm --extra serve

Usage:
    API_KEY=sk-... uv run python examples/dev_workflow_server.py
    # Server starts on http://localhost:8001
    #
    # Stage some changes:   git add src/foo.py
    # Then ask the agent:   "review my staged changes and prepare a commit"
"""

import asyncio
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field

from obelix.adapters.outbound.llm.litellm import LiteLLMProvider
from obelix.core.agent import BaseAgent, SharedMemoryGraph
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.agent.hooks import AgentEvent
from obelix.core.agent.shared_memory import PropagationPolicy
from obelix.core.tool.tool_base import Tool
from obelix.core.tool.tool_decorator import tool
from obelix.core.tracer import Tracer
from obelix.core.tracer.exporters import HTTPExporter
from obelix.infrastructure.logging import setup_logging

load_dotenv()
setup_logging(console_level="INFO")

LITELLM_MODEL = "anthropic/claude-haiku-4-5-20251001"

_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")

# tracer = Tracer(exporter=ConsoleExporter(verbosity=3))

tracer = Tracer(
    exporter=HTTPExporter(endpoint="http://localhost:8100/api/v1/ingest"),
    service_name="DEV_WORKFLOW",
)


# -- Provider ----------------------------------------------------------------


def make_provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model_id=LITELLM_MODEL,
        api_key=os.getenv("API_KEY"),
        max_tokens=8000,
        reasoning_effort="low",
        temperature=1,
    )


# -- Git diff tool (read-only, limited to diff commands) ---------------------


async def _run_git(cmd: str) -> dict:
    """Execute a git command and capture output."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return {
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
        "exit_code": proc.returncode,
    }


@tool(
    name="git_diff",
    description=(
        "Inspect staged git changes. "
        "Use 'summary' for a file list with stats, 'staged' for the full diff, "
        "or 'file' to inspect a single file's staged changes."
    ),
)
class GitDiffTool(Tool):
    """Read-only git diff tool.

    Only diff-reading operations are allowed — no writes, no pushes, no checkouts.
    The LLM chooses a mode; the command is always constructed server-side.
    """

    mode: Literal["summary", "staged", "file"] = Field(
        ...,
        description=(
            "'summary': changed file list with line counts. "
            "'staged': complete diff of all staged changes. "
            "'file': staged diff for a single file path."
        ),
    )
    path: str | None = Field(
        default=None,
        description="File path — required when mode is 'file', ignored otherwise.",
    )

    async def execute(self) -> dict:
        if self.mode == "summary":
            return await _run_git("git diff --staged --stat")

        if self.mode == "staged":
            return await _run_git("git diff --staged")

        if self.mode == "file":
            if not self.path:
                return {"error": "mode='file' requires a non-empty path"}
            # Strip shell metacharacters — the path is only used inside quotes
            safe_path = (
                self.path.replace(";", "")
                .replace("&", "")
                .replace("|", "")
                .replace("`", "")
            )
            return await _run_git(f"git diff --staged -- '{safe_path}'")

        return {"error": f"Unknown mode: {self.mode!r}"}

    def system_prompt_fragment(self) -> str:
        return (
            "\n\n## Git environment\n"
            "Use the `git_diff` tool to inspect staged changes:\n"
            "- `summary` — list of changed files with line stats\n"
            "- `staged`  — full diff of all staged changes\n"
            "- `file`    — diff for a specific file\n\n"
            "This tool is **read-only**: it can inspect diffs but cannot modify, "
            "stage, commit, or push anything."
        )


# -- Agents ------------------------------------------------------------------


class ReviewerAgent(BaseAgent):
    """Reads staged changes and produces a code review + security report."""

    context: str = Field(
        default="",
        description="Optional focus areas or constraints for the review (e.g. 'focus on auth module')",
    )

    def __init__(self, **kwargs):
        super().__init__(
            system_message=(
                "You are a senior code reviewer with a security background.\n"
                "Your job is to review staged git changes and produce two outputs:\n\n"
                "1. **Code review** — invoke the 'code-review' skill (this runs in a "
                "fork: an isolated sub-agent does the deep analysis and returns a "
                "structured report to you).\n"
                "2. **Security check** — invoke the 'security-check' skill for a quick "
                "vulnerability scan.\n\n"
                "Start with `git_diff` in 'summary' mode to see what changed, then "
                "invoke the skills. Present both results clearly in your final answer."
            ),
            provider=make_provider(),
            skills_config=_SKILLS_DIR,
            **kwargs,
        )
        self.register_tool(GitDiffTool())

        # Reject early if git reports nothing staged
        self.on(AgentEvent.BEFORE_LLM_CALL).when(
            lambda s: _nothing_staged(s.agent)
        ).reject(
            "No staged changes found. Run `git add <files>` to stage changes before reviewing."
        )


def _nothing_staged(agent: BaseAgent) -> bool:
    """Return True if the last message indicates there is nothing staged."""
    history = agent.conversation_history
    if len(history) < 2:
        return False
    last_content = getattr(history[-1], "content", "") or ""
    return any(
        phrase in last_content.lower()
        for phrase in (
            "nothing to commit",
            "no changes added",
            "nothing added to commit",
        )
    )


class CommitAgent(BaseAgent):
    """Writes a conventional commit message from review findings."""

    def __init__(self, **kwargs):
        super().__init__(
            system_message=(
                "You are a commit message specialist.\n"
                "You will receive code review findings as context (from shared memory).\n\n"
                "Invoke the 'commit-writer' skill to produce a clean, "
                "conventional commit message that accurately reflects what was changed "
                "and why. Your final answer must be the commit message only — no "
                "additional commentary."
            ),
            provider=make_provider(),
            skills_config=_SKILLS_DIR,
            **kwargs,
        )


class SummaryAgent(BaseAgent):
    """Assembles the final developer report from review + commit findings."""

    def __init__(self, **kwargs):
        super().__init__(
            system_message=(
                "You are a technical writer producing a developer-ready report.\n"
                "You will receive (via shared memory):\n"
                "- Code review findings from the Reviewer Agent\n"
                "- A commit message from the Commit Agent\n\n"
                "Produce a markdown report with exactly this structure:\n\n"
                "## 📋 Code Review\n"
                "[review findings, grouped by severity — Critical / Important / Minor]\n\n"
                "## 🔒 Security\n"
                "[security findings, or 'No issues found']\n\n"
                "## ✅ Suggested Commit\n"
                "```\n[commit message]\n```\n\n"
                "## 🎯 Next Steps\n"
                "[1–3 concrete actions if critical or important issues were found; "
                "otherwise 'Ready to commit.']\n\n"
                "Be concise. Developers scan this before running `git commit`."
            ),
            provider=make_provider(),
            **kwargs,
        )


class CoordinatorAgent(BaseAgent):
    """Orchestrates the dev workflow: review → commit → summary."""

    def __init__(self, **kwargs):
        super().__init__(
            system_message=(
                "You are the Dev Workflow Coordinator.\n"
                "When a developer asks you to review staged changes:\n\n"
                "Step 1 — Call the **Reviewer Agent** to analyze the diff and run code "
                "review + security check skills.\n"
                "Step 2 — Call the **Commit Agent** to generate the commit message "
                "(it receives the review findings automatically via shared memory).\n"
                "Step 3 — Call the **Summary Agent** to produce the final report "
                "(it receives both review and commit message via shared memory).\n\n"
                "RULES:\n"
                "- Never skip steps or change the order.\n"
                "- If a sub-agent REJECTS a request, relay the rejection reason "
                "directly to the developer as your final answer — do not retry.\n"
                "- Keep your own messages brief: the sub-agents do the work."
            ),
            provider=LiteLLMProvider(
                model_id=LITELLM_MODEL,
                api_key=os.getenv("API_KEY"),
                reasoning_effort="medium",
                max_tokens=10_000,
                temperature=1,
            ),
            planning=True,
            **kwargs,
        )


# -- Factory -----------------------------------------------------------------


def create_factory() -> AgentFactory:
    memory_graph = SharedMemoryGraph()
    memory_graph.add_agent("reviewer")
    memory_graph.add_agent("commit_writer")
    memory_graph.add_agent("summary")

    # Reviewer findings flow to both commit_writer and summary
    memory_graph.add_edge(
        "reviewer", "commit_writer", policy=PropagationPolicy.FINAL_RESPONSE_ONLY
    )
    memory_graph.add_edge(
        "reviewer", "summary", policy=PropagationPolicy.FINAL_RESPONSE_ONLY
    )
    # Commit message flows to summary
    memory_graph.add_edge(
        "commit_writer", "summary", policy=PropagationPolicy.FINAL_RESPONSE_ONLY
    )

    factory = AgentFactory()
    factory.with_tracer(tracer)
    factory.with_memory_graph(memory_graph)

    factory.register(
        name="reviewer",
        cls=ReviewerAgent,
        subagent_description=(
            "Analyzes staged git changes and produces a code review + security report. "
            "Pass any developer-provided context (e.g. 'focus on auth') as the query."
        ),
        stateless=True,
    )
    factory.register(
        name="commit_writer",
        cls=CommitAgent,
        subagent_description=(
            "Writes a conventional commit message from the code review findings. "
            "Call this AFTER the reviewer has completed."
        ),
        stateless=True,
    )
    factory.register(
        name="summary",
        cls=SummaryAgent,
        subagent_description=(
            "Assembles the final developer report from review findings and commit message. "
            "Call this LAST, after both reviewer and commit_writer have completed."
        ),
        stateless=True,
    )
    factory.register(name="coordinator", cls=CoordinatorAgent)

    return factory


# -- Serve -------------------------------------------------------------------

if __name__ == "__main__":
    factory = create_factory()
    factory.a2a_serve(
        "coordinator",
        subagents=["reviewer", "commit_writer", "summary"],
        port=8001,
        description=(
            "Dev workflow pipeline: code review + security check + commit message. "
            "Stage your changes with `git add`, then ask: 'review my staged changes'"
        ),
    )
