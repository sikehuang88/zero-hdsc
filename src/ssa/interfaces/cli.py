"""Dependency-free command-line entry point for core installation checks."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from importlib.util import find_spec

from ssa.config import DatabaseConfig, Environment, Settings, load_settings
from ssa.storage.database import Database


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hdsc",
        description="HDSC hyperdimensional space computing console",
    )
    commands = parser.add_subparsers(dest="command")

    for name, help_text in (
        ("doctor", "check the core runtime and database"),
        ("init-db", "initialize or upgrade the database"),
        ("tui", "launch the interactive terminal workspace"),
        ("worker", "run the persistent autonomous lifecycle worker"),
        ("web", "run the HTTP/SSE gateway for browser frontends"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--env",
            choices=[environment.value for environment in Environment],
            default=Environment.DEVELOPMENT.value,
            help="configuration environment (default: development)",
        )
        command.add_argument(
            "--database",
            metavar="PATH",
            help="override the configured SQLite path",
        )
        if name == "tui":
            command.add_argument(
                "--conversation",
                default="cli-primary",
                metavar="ID",
                help="persistent conversation ID (default: cli-primary)",
            )
        if name == "worker":
            command.add_argument(
                "--conversation",
                default="cli-primary",
                metavar="ID",
                help="persistent conversation ID (default: cli-primary)",
            )
            command.add_argument(
                "--once",
                action="store_true",
                help="run one due-job pass and exit",
            )
            command.add_argument(
                "--poll-seconds",
                type=float,
                default=2.0,
                metavar="SECONDS",
                help="continuous worker polling interval (default: 2)",
            )
        if name == "web":
            command.add_argument(
                "--conversation",
                default="web-primary",
                metavar="ID",
                help="persistent conversation ID (default: web-primary)",
            )
            command.add_argument(
                "--host",
                default="127.0.0.1",
                help="bind host (default: 127.0.0.1)",
            )
            command.add_argument(
                "--port",
                type=int,
                default=18787,
                metavar="PORT",
                help="bind port (default: 18787)",
            )
    return parser


def _settings(environment: str, database_path: str | None) -> Settings:
    settings = load_settings(environment)
    if database_path is None:
        return settings
    return settings.model_copy(
        update={
            "database": DatabaseConfig(
                path=database_path,
                busy_timeout_ms=settings.database.busy_timeout_ms,
            )
        }
    )


def _initialize(settings: Settings) -> Database:
    database = Database(settings.database)
    database.initialize()
    return database


def _doctor(settings: Settings) -> int:
    print(
        f"Python:      {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    print(f"Environment: {settings.app.name}")
    database: Database | None = None
    try:
        database = _initialize(settings)
        print(f"Database:    {database.path}")
        print(f"Schema:      v{database.schema_version}")
        print(f"sqlite-vec:  {'loaded' if database.vec_extension_loaded else 'deferred'}")
        print(f"LLM main: {settings.llm.model}")
        print(f"LLM reasoning: {settings.llm.reasoning_model}")
        print(f"LLM endpoint: {settings.llm.base_url}")
        print(
            "LLM thinking:"
            f" {settings.llm.thinking_mode.value}"
            f" (effort={settings.llm.reasoning_effort.value})"
        )
        print(
            f"LiteLLM:     {'installed' if find_spec('litellm') else 'missing (add runtime extra)'}"
        )
        print(
            "LLM secret:  "
            f"{'configured' if settings.secrets.llm_api_key.get_secret_value() else 'not set'}"
        )
        print(f"Multimodal:  {settings.multimodal.model}")
        print(f"MM endpoint: {settings.multimodal.base_url}")
        print(
            "MM secret:   "
            f"{'configured' if settings.secrets.multimodal_api_key.get_secret_value() else 'not set'}"
        )
        print(
            "Win32 tools: "
            f"{'enabled' if settings.win32.enabled and os.name == 'nt' else 'disabled'}"
            f" (clipboard={'on' if settings.win32.clipboard_enabled else 'off'})"
        )
        print("Status:      ok")
        return 0
    except Exception as exc:
        print(f"Status:      error: {exc}", file=sys.stderr)
        return 1
    finally:
        if database is not None:
            database.close()


def _init_database(settings: Settings) -> int:
    database: Database | None = None
    try:
        database = _initialize(settings)
        print(f"Database initialized: {database.path} (schema v{database.schema_version})")
        return 0
    except Exception as exc:
        print(f"Database initialization failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if database is not None:
            database.close()


def _launch_tui(settings: Settings, conversation_id: str) -> int:
    from ssa.interfaces.tui import run_tui

    return run_tui(settings, conversation_id=conversation_id)


def _launch_worker(
    settings: Settings,
    conversation_id: str,
    *,
    once: bool,
    poll_seconds: float,
) -> int:
    from ssa.adapters.deepseek import build_deepseek_adapter
    from ssa.runtime.autonomous import AutonomousRuntime

    if poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    database = _initialize(settings)
    runtime = AutonomousRuntime(
        database=database,
        llm=build_deepseek_adapter(settings),
        settings=settings,
        conversation_id=conversation_id,
    )

    async def run() -> None:
        jobs = runtime.bootstrap()
        print(f"Lifecycle ready: {len(jobs)} persistent jobs for {conversation_id}")
        while True:
            result = await runtime.tick()
            tick = result.lifecycle
            print(
                "Lifecycle tick: "
                f"leased={tick.leased} rescheduled={tick.rescheduled} "
                f"failed={tick.failed} proactive={len(result.proactive_events)}"
            )
            if once:
                return
            await asyncio.sleep(poll_seconds)

    try:
        asyncio.run(run())
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        database.close()


def _launch_web(
    settings: Settings,
    conversation_id: str,
    *,
    host: str,
    port: int,
) -> int:
    from ssa.interfaces import web_gateway

    app = web_gateway.create_app(settings, conversation_id=conversation_id)
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the HDSC CLI and return a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    settings = _settings(str(args.env), args.database)
    if args.command == "doctor":
        return _doctor(settings)
    if args.command == "init-db":
        return _init_database(settings)
    if args.command == "tui":
        try:
            return _launch_tui(settings, str(args.conversation))
        except Exception as exc:
            print(f"TUI startup failed: {exc}", file=sys.stderr)
            return 1
    if args.command == "worker":
        try:
            return _launch_worker(
                settings,
                str(args.conversation),
                once=bool(args.once),
                poll_seconds=float(args.poll_seconds),
            )
        except Exception as exc:
            print(f"Worker startup failed: {exc}", file=sys.stderr)
            return 1
    if args.command == "web":
        try:
            return _launch_web(
                settings,
                str(args.conversation),
                host=str(args.host),
                port=int(args.port),
            )
        except ModuleNotFoundError as exc:
            print(
                f"Web gateway dependency missing: {exc}\n"
                "Install with: uv sync --extra web",
                file=sys.stderr,
            )
            return 1
        except Exception as exc:
            print(f"Web gateway startup failed: {exc}", file=sys.stderr)
            return 1
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
