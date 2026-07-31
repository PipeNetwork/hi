"""Harbor agent adapter running `hi` on Terminal-Bench.

The adapter uploads a locally built static Linux binary (see build-linux.sh)
into each task container and invokes hi's one-shot mode with the task
instruction. Harbor then scores the resulting container state with the task's
own tests, so hi's built-in verifier is advisory here: `--allow-unverified`
keeps a completed-but-unverified turn at exit 0 instead of surfacing as an
agent crash.
"""

import os
import re
import shlex
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.base import (
    ApiError,
    BaseInstalledAgent,
    NonZeroAgentExitCodeError,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# Mirrors `ProviderName::key_envs` in crates/hi-cli/src/config/cli.rs — the env
# vars hi itself reads per provider, in its own precedence order.
PROVIDER_KEY_ENVS: dict[str, list[str]] = {
    "anthropic": ["HI_API_KEY", "ANTHROPIC_API_KEY"],
    "openai": ["HI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"],
    "pipenetwork": ["PIPENETWORK_API_KEY", "HI_API_KEY", "OPENAI_API_KEY"],
    "xai": ["XAI_API_KEY", "HI_API_KEY"],
    "ollama": ["HI_API_KEY", "OLLAMA_API_KEY"],
}

_CONTAINER_BINARY = "/usr/local/bin/hi"
_OUTPUT_FILE = "/installed-agent/hi-output.txt"
# Harbor's default per-task agent cap, used when the task definition can't be read.
_DEFAULT_AGENT_TIMEOUT_SEC = 900.0


def _default_binary_path() -> Path:
    return Path(__file__).resolve().parents[2] / "dist" / "hi-linux"


class HiAgent(BaseInstalledAgent):
    """Runs the hi coding agent in one-shot mode inside the task container."""

    def __init__(self, *args: Any, turn_deadline_sec: int | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._deadline_override = (
            int(turn_deadline_sec) if turn_deadline_sec is not None else None
        )

    @staticmethod
    @override
    def name() -> str:
        return "hi"

    @override
    def get_version_command(self) -> str | None:
        return f"{_CONTAINER_BINARY} --version"

    def _turn_deadline_sec(self) -> int:
        """Seconds hi may spend before it must stop and settle.

        Harbor's per-task agent cap is not handed to custom agents (only the
        oracle gets it), so it is read from the task definition Harbor already
        cached on disk; the trial directory is named `<task>__<id>`. Falls back
        to Harbor's 900s task default. `--ak turn_deadline_sec=N` overrides.
        """
        if self._deadline_override is not None:
            return self._deadline_override
        timeout = _DEFAULT_AGENT_TIMEOUT_SEC
        task_name = self.logs_dir.parent.name.rsplit("__", 1)[0]
        if task_name:
            for task_toml in Path(
                os.path.expanduser("~/.cache/harbor/tasks")
            ).glob(f"*/{task_name}/task.toml"):
                match = re.search(
                    r"^\s*timeout_sec\s*=\s*([0-9.]+)",
                    task_toml.read_text(),
                    re.MULTILINE,
                )
                if match:
                    timeout = float(match.group(1))
                break
        # Leave headroom for the wind-down (a last verification pass, workspace
        # reconciliation, the report) so it finishes inside the cap. Reserve an
        # absolute floor as well as a share: settle cost is dominated by the
        # task's own test suite, which does not shrink with a smaller cap.
        reserve = max(150.0, timeout * 0.15)
        return max(60, int(timeout - reserve))

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        binary = Path(os.environ.get("HI_TB_BINARY", _default_binary_path()))
        if not binary.is_file():
            raise FileNotFoundError(
                f"hi Linux binary not found at {binary}. "
                "Run bench/terminal-bench/build-linux.sh first, or point "
                "HI_TB_BINARY at a static Linux build of hi."
            )
        await environment.upload_file(binary, _CONTAINER_BINARY)
        await self.exec_as_root(
            environment,
            f"chmod 755 {_CONTAINER_BINARY} && {_CONTAINER_BINARY} --version",
        )

    @with_prompt_template
    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError(
                "model must be provider/model (e.g. anthropic/claude-opus-4-8); "
                f"got {self.model_name!r}"
            )
        provider, model = self.model_name.split("/", 1)
        key_envs = PROVIDER_KEY_ENVS.get(provider)
        if key_envs is None:
            raise ValueError(
                f"unknown provider {provider!r}; hi supports: "
                + ", ".join(sorted(PROVIDER_KEY_ENVS))
            )

        env: dict[str, str] = {}
        for key in key_envs:
            value = self._get_env(key)
            if value:
                env[key] = value
        if not env and provider != "ollama":
            raise ValueError(
                f"no API key for provider {provider!r}; set one of: "
                + ", ".join(key_envs)
            )

        # --no-save/--no-memory: nothing should persist across tasks;
        # --allow-unverified: the benchmark's own tests are the verifier;
        # --keep-background: several tasks require a service still listening
        # when the verifier runs ("keep it running in the background"), and it
        # probes within a second of hi exiting;
        # --turn-deadline: stop and settle before Harbor's agent cap kills the
        # process mid-edit, which leaves the reward to whatever was on disk.
        deadline = self._turn_deadline_sec()
        command = (
            f"{_CONTAINER_BINARY}"
            f" --provider {shlex.quote(provider)}"
            f" --model {shlex.quote(model)}"
            " --no-save --no-memory --allow-unverified --keep-background"
            f" --turn-deadline {deadline}"
            f" {shlex.quote(instruction)}"
            f" 2>&1 | tee {_OUTPUT_FILE}"
        )
        try:
            await self.exec_as_agent(environment, command, env=env)
        except ApiError:
            # Provider-side failures (rate limits, 500s) stay fatal so Harbor
            # retry policy (--retry-include ApiRateLimitError) can target them.
            raise
        except NonZeroAgentExitCodeError as error:
            # hi's one-shot exit code encodes its own turn status — e.g. a
            # stalled internal verify loop exits 1 with the work complete
            # (observed: pypi-server scored reward=1 under exit 1). The task's
            # verifier is ground truth here; log and let it score the state.
            self.logger.warning(
                "hi exited non-zero; deferring to the task verifier: %s", error
            )
