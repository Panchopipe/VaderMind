"""CLI entrypoint for VaderMind."""

from vadermind.telemetry import STARTUP

STARTUP.mark("main")  # before the heavy imports, to time them at start-up

from vadermind.orchestrator import run_cli  # noqa: E402


if __name__ == "__main__":
    run_cli()
