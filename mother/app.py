"""
Mother — root Textual application.

Entry point for the Mother TUI. Handles first-run detection,
screen routing, and global keybindings.
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from mother import __version__
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Static, Footer
from textual.containers import Container

logger = logging.getLogger("mother.app")

_DEFAULT_PID_PATH = Path.home() / ".motherlabs" / "mother.pid"


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still alive.
        return True


def acquire_lock(pid_path: Path | None = None) -> bool:
    """Write PID lockfile atomically. Returns True if lock acquired.

    Uses atomic write (temp file + rename) to prevent TOCTOU race.
    Sets 0600 permissions on the lockfile.
    """
    pid_path = pid_path or _DEFAULT_PID_PATH
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    if pid_path.exists():
        try:
            stored_pid = int(pid_path.read_text().strip())
            if _is_pid_alive(stored_pid):
                return False
        except (ValueError, OSError):
            pass  # Stale/corrupt file — overwrite it.

    # Atomic write: write to temp file, then rename
    import tempfile
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(pid_path.parent), prefix=".mother_lock_"
        )
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, str(pid_path))
    except OSError:
        # Fallback: direct write if atomic fails
        pid_path.write_text(str(os.getpid()))
        try:
            os.chmod(str(pid_path), 0o600)
        except OSError:
            pass
    return True


def release_lock(pid_path: Path | None = None) -> None:
    """Remove PID lockfile."""
    pid_path = pid_path or _DEFAULT_PID_PATH
    try:
        if pid_path.exists():
            stored_pid = int(pid_path.read_text().strip())
            if stored_pid == os.getpid():
                pid_path.unlink()
    except (ValueError, OSError):
        pass


class LoadingScreen(Screen):
    """Brief loading screen shown during initialization."""

    def compose(self) -> ComposeResult:
        with Container(id="loading-container"):
            yield Static("Starting Mother...", id="loading-text")


class MotherApp(App):
    """The Mother TUI application."""

    TITLE = "Mother"
    CSS_PATH = "mother.tcss"  # Default, can be overridden in __init__

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+comma", "open_settings", "Settings", show=True),
        Binding("ctrl+r", "restart", "Restart", show=False),
    ]

    def __init__(self, config_path: str | None = None):
        super().__init__()
        self._config_path = config_path

        # Load theme from config
        from mother.config import load_config
        config = load_config(config_path)
        theme = config.theme if hasattr(config, "theme") else "default"

        # Set CSS path based on theme
        if theme == "alien":
            self.CSS_PATH = "mother_alien.tcss"
        else:
            self.CSS_PATH = "mother.tcss"

    def on_mount(self) -> None:
        """Route to setup or chat based on first-run detection."""
        try:
            from mother.config import detect_first_run, load_config

            if detect_first_run(self._config_path):
                from mother.screens.setup import SetupScreen
                self.push_screen(SetupScreen(config_path=self._config_path))
            else:
                from mother.screens.chat import ChatScreen
                config = load_config(self._config_path)
                self.push_screen(ChatScreen(config=config, config_path=self._config_path))
        except Exception as e:
            logger.error(f"Startup error: {e}")
            self.push_screen(LoadingScreen())

    def action_open_settings(self) -> None:
        """Open the settings screen."""
        from mother.screens.settings import SettingsScreen
        self.push_screen(SettingsScreen(config_path=self._config_path))

    def action_quit(self) -> None:
        """Quit the application with death protocol — save handoff + journal entry."""
        try:
            from pathlib import Path as _Path
            handoff_path = _Path.home() / ".motherlabs" / "last_handoff.md"
            handoff_path.parent.mkdir(parents=True, exist_ok=True)
            screen = self.screen
            bridge = getattr(screen, '_bridge', None)
            db_path = getattr(getattr(screen, '_store', None), '_path', None)
            if bridge:
                handoff_md = bridge.generate_handoff(db_path=db_path)
                handoff_path.write_text(handoff_md)
                if db_path:
                    try:
                        from mother.journal import BuildJournal, JournalEntry
                        journal = BuildJournal(db_path)
                        journal.record(JournalEntry(
                            event_type="session_end",
                            description="Graceful shutdown. Handoff saved.",
                            success=True, domain="system",
                        ))
                        journal.close()
                    except Exception:
                        pass
        except Exception:
            pass
        self.exit()

    def action_restart(self) -> None:
        """Restart the application by re-exec'ing the process."""
        # Flush all buffers before exec
        sys.stdout.flush()
        sys.stderr.flush()
        release_lock()
        try:
            os.execv(sys.executable, [sys.executable, "-m", "mother.app"] + sys.argv[1:])
        except OSError:
            # exec failed — re-acquire lock so we're still protected
            acquire_lock()
            raise


async def _daemon_main(bridge, daemon):
    """Async entry point for headless daemon. Runs until signalled."""
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(_shutdown(daemon, stop_event)))

    await daemon.start()
    logger.info("Daemon mode started — autonomous self-improvement active")

    # Social queue processing in background
    async def _social_loop():
        while not stop_event.is_set():
            try:
                await asyncio.sleep(60)
                if stop_event.is_set():
                    break
                await bridge.process_social_queue()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"Social queue processing skipped: {e}")

    social_task = asyncio.create_task(_social_loop())

    try:
        await stop_event.wait()
    finally:
        social_task.cancel()
        try:
            await social_task
        except asyncio.CancelledError:
            pass
        await daemon.stop()
        logger.info("Daemon shutdown complete")


async def _shutdown(daemon, stop_event):
    """Signal handler — trigger clean shutdown."""
    logger.info("Received shutdown signal")
    stop_event.set()


def _run_daemon(config_path=None):
    """Run Mother in headless daemon mode (no TUI)."""
    import asyncio
    from logging.handlers import RotatingFileHandler

    from mother.config import load_config
    from mother.bridge import EngineBridge
    from mother.daemon import DaemonMode, DaemonConfig

    config = load_config(config_path)

    # File logging
    log_dir = Path.home() / ".motherlabs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.daemon_log_file or str(log_dir / "daemon.log")
    handler = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.info("Starting Mother daemon")

    # Inject config API keys into os.environ (same as chat.py on_mount)
    _API_KEY_ENV_MAP = {
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "grok": "XAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    for _prov, _env_var in _API_KEY_ENV_MAP.items():
        _key_val = config.api_keys.get(_prov)
        if _key_val and not os.environ.get(_env_var):
            os.environ[_env_var] = _key_val

    # Create bridge
    api_key = config.api_keys.get(config.provider)
    openai_key = config.api_keys.get("openai")
    bridge = EngineBridge(
        provider=config.provider,
        model=config.get_model(),
        api_key=api_key,
        file_access=config.file_access,
        openai_api_key=openai_key,
        pipeline_mode=getattr(config, "pipeline_mode", "staged"),
    )

    # Repo dir for self-build
    repo_dir = str(Path(__file__).resolve().parent.parent)

    # compile_fn: called by DaemonMode scheduler when a goal is picked
    async def _compile_fn(input_text, domain="software"):
        result = await bridge.self_build(
            prompt=input_text,
            repo_dir=repo_dir,
        )
        if result.get("success"):
            logger.info("Self-build succeeded — restarting daemon to load changes")
            # Process social queue before restart
            try:
                await bridge.process_social_queue()
            except Exception as e:
                logger.debug(f"Pre-restart social queue skipped: {e}")
            # Flush buffers before exec
            sys.stdout.flush()
            sys.stderr.flush()
            release_lock()
            try:
                os.execv(sys.executable, [sys.executable, "-m", "mother.app", "--daemon"])
            except OSError:
                acquire_lock()
                logger.error("Daemon restart failed — re-acquired lock")
                raise
        return result

    # Create daemon
    daemon_config = DaemonConfig(
        health_check_interval=config.daemon_health_check_interval,
        max_queue_size=config.daemon_max_queue_size,
        auto_heal=config.daemon_auto_heal,
        idle_shutdown_hours=config.daemon_idle_shutdown_hours,
        log_file=log_path,
    )
    daemon = DaemonMode(
        config=daemon_config,
        config_dir=log_dir,
        compile_fn=_compile_fn,
    )

    # Wire weekly governance
    if getattr(config, "weekly_build_enabled", True):
        daemon.configure_weekly_build(
            enabled=True,
            briefing_day=getattr(config, "weekly_briefing_day", 6),
            briefing_hour=getattr(config, "weekly_briefing_hour", 10),
            window_start=getattr(config, "build_window_start_hour", 22),
            window_end=getattr(config, "build_window_end_hour", 6),
            window_day=getattr(config, "build_window_day", 6),
            max_per_window=getattr(config, "build_max_per_window", 10),
        )

    asyncio.run(_daemon_main(bridge, daemon))


def main():
    """Entry point for the `mother` command."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mother",
        description="Mother — local AI development environment.",
    )
    parser.add_argument(
        "--version", action="version", version=f"Mother {__version__} (Motherlabs)"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file"
    )
    parser.add_argument(
        "--daemon", action="store_true", help="Run headless (no TUI)"
    )
    args = parser.parse_args()

    if not acquire_lock():
        print("Mother is already running.", file=sys.stderr)
        sys.exit(1)

    try:
        if args.daemon:
            _run_daemon(config_path=args.config)
        else:
            app = MotherApp(config_path=args.config)
            app.run()
    finally:
        release_lock()


if __name__ == "__main__":
    main()
