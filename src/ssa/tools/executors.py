"""Host-level PowerShell and file capabilities for the tool kernel."""

from __future__ import annotations

import asyncio
import base64
import locale
import logging
import os
import shutil
import subprocess
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from ssa.config import (
    ExternalTruthConfig,
    FirecrawlConfig,
    MultimodalConfig,
    OpencodeConfig,
    ToolConfig,
    Win32Config,
)
from ssa.services.prediction_service import GroundedPredictionService
from ssa.tools.coding import CodingToolExecutor
from ssa.tools.external_truth import ExternalTruthExecutor
from ssa.tools.firecrawl import FirecrawlExecutor
from ssa.tools.gpt_bridge import GPTBridgeExecutor
from ssa.tools.image_generation import ImageGenerationExecutor
from ssa.tools.kernel import ToolKernel
from ssa.tools.mineradio import MineradioExecutor
from ssa.tools.models import (
    PowerShellArguments,
    ReadFileArguments,
    ToolAutonomyContext,
    ToolOutcome,
    WriteFileArguments,
)
from ssa.tools.predictions import GroundedPredictionExecutor
from ssa.tools.registry import ToolCapability, ToolRegistry
from ssa.tools.web_browser import BrowserSubAgent
from ssa.tools.win32_registry import Win32ToolExecutor

logger = logging.getLogger(__name__)


class GlobalToolExecutor:
    """Executes against the host filesystem without a workspace-root constraint."""

    def __init__(self, config: ToolConfig) -> None:
        self._config = config

    def register_into(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolCapability(
                name="powershell",
                description=(
                    "Run a PowerShell command on the global Windows host. The working directory "
                    "may be any host path. Set elevated=true to relaunch through Windows RunAs "
                    "when administrator rights are required."
                ),
                arguments_model=PowerShellArguments,
                handler=self.run_powershell,
            )
        )
        registry.register(
            ToolCapability(
                name="read_file",
                description=(
                    "Read bytes from any host file path without a workspace boundary. Returns "
                    "decoded text by default or Base64 when as_base64=true."
                ),
                arguments_model=ReadFileArguments,
                handler=self.read_file,
            )
        )
        registry.register(
            ToolCapability(
                name="write_file",
                description=(
                    "Write, append, or exclusively create a file at any host path without a "
                    "workspace boundary. Content may be plain text or Base64."
                ),
                arguments_model=WriteFileArguments,
                handler=self.write_file,
            )
        )

    async def run_powershell(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = PowerShellArguments.model_validate(
            {"timeout_seconds": self._config.default_timeout_seconds, **raw_arguments}
        )
        if arguments.elevated:
            if not self._config.allow_elevation:
                return ToolOutcome(ok=False, error="elevated PowerShell is disabled")
            return await self._run_elevated(arguments)
        return await self._run_direct(arguments)

    async def read_file(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = ReadFileArguments.model_validate(raw_arguments)
        path = _global_path(arguments.path)
        with path.open("rb") as handle:
            handle.seek(arguments.offset_bytes)
            payload = handle.read(arguments.max_bytes + 1)
        truncated = len(payload) > arguments.max_bytes
        payload = payload[: arguments.max_bytes]
        output = (
            base64.b64encode(payload).decode("ascii")
            if arguments.as_base64
            else payload.decode(arguments.encoding, errors="replace")
        )
        return ToolOutcome(
            ok=True,
            output=output,
            metadata={
                "path": str(path),
                "bytes_read": len(payload),
                "offset_bytes": arguments.offset_bytes,
                "truncated": truncated,
                "base64": arguments.as_base64,
            },
        )

    async def write_file(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = WriteFileArguments.model_validate(raw_arguments)
        path = _global_path(arguments.path)
        if arguments.create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            base64.b64decode(arguments.content, validate=True)
            if arguments.content_is_base64
            else arguments.content.encode(arguments.encoding)
        )
        mode = {"overwrite": "wb", "append": "ab", "create": "xb"}[arguments.mode]
        with path.open(mode) as handle:
            handle.write(payload)
        return ToolOutcome(
            ok=True,
            output=f"wrote {len(payload)} bytes to {path}",
            metadata={
                "path": str(path),
                "bytes_written": len(payload),
                "mode": arguments.mode,
                "base64": arguments.content_is_base64,
            },
        )

    async def _run_direct(self, arguments: PowerShellArguments) -> ToolOutcome:
        process = await self._spawn(
            self._config.powershell_executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "$OutputEncoding = [Console]::OutputEncoding = "
            "[Text.UTF8Encoding]::new($false); " + arguments.command,
            cwd=arguments.working_directory,
        )
        stdout, stderr, timed_out = await _communicate(process, arguments.timeout_seconds)
        if timed_out:
            return ToolOutcome(ok=False, error="PowerShell command timed out")
        output = _decode_process_output(stdout)
        error = _decode_process_output(stderr).strip() or None
        return ToolOutcome(
            ok=process.returncode == 0,
            output=output,
            error=error,
            exit_code=process.returncode,
            elevated=False,
            metadata={"working_directory": arguments.working_directory},
        )

    async def _run_elevated(self, arguments: PowerShellArguments) -> ToolOutcome:
        if os.name != "nt":
            return ToolOutcome(ok=False, error="RunAs elevation requires Windows")

        temp = Path(tempfile.mkdtemp(prefix="hdsc-tool-"))
        try:
            command_path = temp / "command.ps1"
            runner_path = temp / "runner.ps1"
            output_path = temp / "output.txt"
            exit_path = temp / "exit.txt"
            command_path.write_text(arguments.command, encoding="utf-8")
            runner_path.write_text(
                self._elevated_runner_script(
                    command_path,
                    output_path,
                    exit_path,
                    arguments.working_directory,
                ),
                encoding="utf-8",
            )
            launcher = self._elevation_launcher_command(
                self._config.powershell_executable,
                runner_path,
            )
            process = await self._spawn(
                self._config.powershell_executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                launcher,
            )
            stdout, stderr, timed_out = await _communicate(process, arguments.timeout_seconds)
            if timed_out:
                return ToolOutcome(ok=False, error="elevated PowerShell command timed out")
            child_output = (
                output_path.read_text(encoding="utf-8-sig", errors="replace")
                if output_path.exists()
                else _decode_process_output(stdout)
            )
            exit_code = (
                int(exit_path.read_text(encoding="ascii").strip())
                if exit_path.exists()
                else process.returncode
            )
            error = _decode_process_output(stderr).strip() or None
            return ToolOutcome(
                ok=exit_code == 0,
                output=child_output,
                error=error,
                exit_code=exit_code,
                elevated=True,
                metadata={"working_directory": arguments.working_directory},
            )
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    async def _spawn(
        self,
        *command: str,
        cwd: str | None = None,
    ) -> asyncio.subprocess.Process:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creationflags,
        )

    @staticmethod
    def _elevated_runner_script(
        command_path: Path,
        output_path: Path,
        exit_path: Path,
        working_directory: str | None,
    ) -> str:
        location = (
            f"Set-Location -LiteralPath '{_quote_ps_literal(working_directory)}'"
            if working_directory is not None
            else ""
        )
        return "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                "$ProgressPreference = 'SilentlyContinue'",
                location,
                "$exitCode = 0",
                "try {",
                f"  & '{_quote_ps_literal(str(command_path))}' *>&1 | "
                f"Out-File -LiteralPath '{_quote_ps_literal(str(output_path))}' -Encoding utf8",
                "  if ($null -ne $LASTEXITCODE) { $exitCode = $LASTEXITCODE }",
                "} catch {",
                f"  $_ | Out-File -LiteralPath '{_quote_ps_literal(str(output_path))}' "
                "-Encoding utf8 -Append",
                "  $exitCode = 1",
                "}",
                f"Set-Content -LiteralPath '{_quote_ps_literal(str(exit_path))}' "
                "-Value $exitCode -Encoding ascii",
                "exit $exitCode",
            ]
        )

    @staticmethod
    def _elevation_launcher_command(executable: str, runner_path: Path) -> str:
        executable_literal = _quote_ps_literal(executable)
        runner_literal = str(runner_path).replace('"', '`"')
        return (
            f"$argsLine = '-NoLogo -NoProfile -ExecutionPolicy Bypass -File \"{runner_literal}\"'; "
            f"$p = Start-Process -FilePath '{executable_literal}' -ArgumentList $argsLine "
            "-Verb RunAs -Wait -PassThru; exit $p.ExitCode"
        )


async def _communicate(
    process: asyncio.subprocess.Process,
    timeout_seconds: int,
) -> tuple[bytes, bytes, bool]:
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        return stdout, stderr, False
    except TimeoutError:
        process.kill()
        stdout, stderr = await process.communicate()
        return stdout, stderr, True


def _decode_process_output(payload: bytes) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        pass
    encoding = locale.getpreferredencoding(False) or "utf-8"
    return payload.decode(encoding, errors="replace")


def _global_path(raw_path: str) -> Path:
    return Path(raw_path).expanduser().resolve(strict=False)


def _quote_ps_literal(value: str) -> str:
    return value.replace("'", "''")


def build_default_tool_kernel(
    config: ToolConfig,
    external_truth_config: ExternalTruthConfig | None = None,
    multimodal_config: MultimodalConfig | None = None,
    multimodal_api_key: str = "",
    win32_config: Win32Config | None = None,
    firecrawl_config: FirecrawlConfig | None = None,
    firecrawl_api_key: str = "",
    prediction_service: GroundedPredictionService | None = None,
    opencode_config: OpencodeConfig | None = None,
    opencode_server_password: str = "",
    opencode_server_username: str = "opencode",
) -> ToolKernel:
    registry = ToolRegistry()
    GlobalToolExecutor(config).register_into(registry)
    CodingToolExecutor().register_into(registry)
    ImageGenerationExecutor().register_into(registry)
    BrowserSubAgent().register_into(registry)
    if prediction_service is not None:
        GroundedPredictionExecutor(prediction_service).register_into(registry)
    truth_config = external_truth_config or ExternalTruthConfig()
    if truth_config.enabled:
        ExternalTruthExecutor(truth_config).register_into(registry)
    if firecrawl_config is not None and firecrawl_config.enabled and firecrawl_api_key:
        FirecrawlExecutor(firecrawl_config, firecrawl_api_key).register_into(registry)
    if multimodal_config is not None and multimodal_config.enabled and multimodal_api_key:
        GPTBridgeExecutor(multimodal_config, multimodal_api_key).register_into(registry)
    if opencode_config is not None and opencode_config.enabled and opencode_server_password.strip():
        from ssa.tools.opencode import OpencodeExecutor

        OpencodeExecutor(
            opencode_config,
            opencode_server_password,
            server_username=opencode_server_username,
        ).register_into(registry)
    if os.name == "nt":
        MineradioExecutor().register_into(registry)
        if find_spec("websockets") is None:
            logger.info("soda_music tools unavailable: websockets not installed")
        else:
            try:
                from ssa.tools.soda_music import SodaMusicExecutor
            except ImportError as exc:
                logger.info("soda_music tools unavailable: %s", exc)
            else:
                SodaMusicExecutor().register_into(registry)
        if win32_config is not None and win32_config.enabled:
            Win32ToolExecutor(win32_config).register_into(registry)
    return ToolKernel(registry, max_output_chars=config.max_output_chars)


__all__ = ["GlobalToolExecutor", "build_default_tool_kernel"]
