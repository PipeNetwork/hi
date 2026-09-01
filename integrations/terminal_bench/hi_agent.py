"""Harbor (Terminal-Bench 2.0) installed-agent adapter for `hi`.

Harbor spins up each task's Docker environment, calls ``install()`` to place
the agent inside it, runs the task instruction through ``run()``, then scores
the resulting container state with the task's own tests (outcome-only, like
the Terminal-Bench paper protocol). This adapter uploads a Linux build of
``hi`` (see docs/terminal-bench.md for producing one), runs it one-shot with
``--report``, and surfaces hi's token usage and trajectory telemetry back to
Harbor.

Usage::

    export HI_AGENT_BINARY=/path/to/linux/hi
    export PIPENETWORK_API_KEY=...
    harbor run -d terminal-bench@2.0 \
        --agent integrations.terminal_bench.hi_agent:HiAgent \
        -m pipenetwork/ipop/coder-balanced -n 4
"""

import json
import os
import shlex
from pathlib import Path
from typing import override

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_REPORT = "/logs/agent/hi-report.json"
_OUTPUT = "/logs/agent/hi.txt"

# Harbor model prefix (-m "<provider>/<model>") → (hi --provider, key sources).
_PROVIDERS = {
    "pipenetwork": ("pipenetwork", ["PIPENETWORK_API_KEY", "HI_API_KEY"]),
    "anthropic": ("anthropic", ["ANTHROPIC_API_KEY", "HI_API_KEY"]),
    "openrouter": ("openai", ["OPENROUTER_API_KEY", "HI_API_KEY"]),
    "openai": ("openai", ["OPENAI_API_KEY", "HI_API_KEY"]),
}


class HiAgent(BaseInstalledAgent):
    """Container-installed `hi` driven one-shot per task instruction."""

    @staticmethod
    @override
    def name() -> str:
        return "hi"

    @override
    def get_version_command(self) -> str | None:
        return "hi --version"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        binary = os.environ.get("HI_AGENT_BINARY", "")
        if not binary or not Path(binary).is_file():
            raise ValueError(
                "HI_AGENT_BINARY must point at a Linux build of `hi` matching "
                "the task image architecture — see docs/terminal-bench.md"
            )
        await self.exec_as_root(environment, command="mkdir -p /opt/hi")
        await environment.upload_file(binary, "/opt/hi/hi")
        await self.exec_as_root(
            environment,
            command=(
                "chmod 755 /opt/hi/hi && ln -sf /opt/hi/hi /usr/local/bin/hi "
                "&& hi --version"
            ),
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError(
                "pass -m <provider>/<model>, e.g. pipenetwork/ipop/coder-balanced"
            )
        provider, model = self.model_name.split("/", 1)
        if provider not in _PROVIDERS:
            raise ValueError(
                f"unknown provider {provider!r}; known: {sorted(_PROVIDERS)}"
            )
        hi_provider, key_sources = _PROVIDERS[provider]

        env: dict[str, str] = {}
        for key in key_sources:
            value = os.environ.get(key)
            if value:
                env["HI_API_KEY"] = value
                break
        # Model routing comes entirely from flags/env — never let a config
        # file baked into the task image shadow it.
        env["XDG_CONFIG_HOME"] = "/tmp/hi-xdg"

        # `|| true`: Terminal-Bench scoring is outcome-only. An unverified or
        # incomplete hi turn exits nonzero, but the task tests — not the exit
        # code — decide the trial, so the verifier must always get to run.
        command = (
            "{ hi --plain --no-save --allow-unverified "
            f"--provider {hi_provider} -m {shlex.quote(model)} "
            f"--report {_REPORT} "
            f"{shlex.quote(instruction)} || true; }} "
            f"2>&1 </dev/null | tee {_OUTPUT}"
        )
        await self.exec_as_agent(environment, command=command, env=env)

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        report_path = self.logs_dir / "hi-report.json"
        if not report_path.exists():
            return
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        # Schema v2 nests usage as {"session": {...}, "turn": {...}}; older
        # reports carried flat top-level token fields.
        usage = report.get("usage") or {}
        session = usage.get("session") or usage
        context.n_input_tokens = session.get("input_tokens") or report.get("input_tokens")
        context.n_cache_tokens = session.get("cache_read_tokens")
        context.n_output_tokens = session.get("output_tokens")
        context.metadata = {
            "hi_outcome": report.get("outcome"),
            "hi_telemetry": report.get("telemetry"),
            "hi_verification": report.get("verification"),
        }
