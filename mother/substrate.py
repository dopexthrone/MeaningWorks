"""
Substrate portability — detect and abstract platform differences.

LEAF module (stdlib only). Mother's capabilities degrade gracefully
on non-macOS platforms instead of crashing on missing commands.

Genome #13: substrate-portable — can migrate between infrastructure
without losing her compiled self.
"""

import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("mother.substrate")

# Platform constants
DARWIN = "darwin"
LINUX = "linux"
WIN32 = "win32"


@dataclass(frozen=True)
class SubstrateCapabilities:
    """What the current platform can do. Frozen — computed once at startup."""

    platform: str               # "darwin", "linux", "win32"
    has_spotlight: bool         # mdfind available
    has_fsevents: bool          # filesystem watch native
    has_say: bool               # TTS via `say` command
    open_command: str           # "open" / "xdg-open" / "start"
    home_dir: Path
    config_dir: Path            # ~/.motherlabs
    temp_dir: Path
    has_notify: bool            # can send desktop notifications
    python_version: str         # e.g. "3.14.0"


def _command_exists(name: str) -> bool:
    """Check if a command is available on PATH."""
    return shutil.which(name) is not None


class SubstrateDetector:
    """Detect platform capabilities and provide cross-platform operations."""

    _cached: Optional[SubstrateCapabilities] = None

    @classmethod
    def detect(cls) -> SubstrateCapabilities:
        """Detect platform capabilities. Cached after first call."""
        if cls._cached is not None:
            return cls._cached

        plat = sys.platform
        home = Path.home()
        config_dir = home / ".motherlabs"
        temp_dir = Path(tempfile.gettempdir())

        # Platform-specific open command
        if plat == DARWIN:
            open_cmd = "open"
        elif plat == LINUX:
            open_cmd = "xdg-open"
        elif plat == WIN32:
            open_cmd = "start"
        else:
            open_cmd = ""

        # Notification capability
        if plat == DARWIN:
            has_notify = _command_exists("osascript")
        elif plat == LINUX:
            has_notify = _command_exists("notify-send")
        else:
            has_notify = False

        caps = SubstrateCapabilities(
            platform=plat,
            has_spotlight=(plat == DARWIN and _command_exists("mdfind")),
            has_fsevents=(plat == DARWIN),
            has_say=(plat == DARWIN and _command_exists("say")),
            open_command=open_cmd,
            home_dir=home,
            config_dir=config_dir,
            temp_dir=temp_dir,
            has_notify=has_notify,
            python_version=platform.python_version(),
        )
        cls._cached = caps
        return caps

    @classmethod
    def reset_cache(cls) -> None:
        """Clear cached capabilities. For testing."""
        cls._cached = None

    @staticmethod
    def find_files(
        query: str,
        directory: Optional[Path] = None,
        max_results: int = 20,
        timeout: int = 10,
    ) -> List[Path]:
        """Platform-adaptive file search. mdfind on macOS, fallback to glob."""
        caps = SubstrateDetector.detect()

        if caps.has_spotlight:
            results = _find_spotlight(query, directory, max_results, timeout)
            if results is not None:
                return results
            # Spotlight failed, fall through to glob
            logger.debug("Spotlight search failed, falling back to glob")

        return _find_glob(query, directory, max_results)

    @staticmethod
    def open_file(path: Path) -> bool:
        """Platform-adaptive file open. Returns True on success."""
        caps = SubstrateDetector.detect()

        if not caps.open_command:
            logger.warning(f"No open command for platform {caps.platform}")
            return False

        if not path.exists():
            logger.warning(f"File not found: {path}")
            return False

        try:
            if caps.platform == WIN32:
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(
                    [caps.open_command, str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return True
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(f"Failed to open {path}: {e}")
            return False

    @staticmethod
    def notify(title: str, message: str) -> bool:
        """Platform-adaptive notification. Returns True on success."""
        caps = SubstrateDetector.detect()

        if not caps.has_notify:
            logger.debug(f"Notifications not available on {caps.platform}")
            return False

        try:
            if caps.platform == DARWIN:
                script = (
                    f'display notification "{message}" '
                    f'with title "{title}"'
                )
                subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    timeout=5,
                )
                return True
            elif caps.platform == LINUX:
                subprocess.run(
                    ["notify-send", title, message],
                    capture_output=True,
                    timeout=5,
                )
                return True
        except (subprocess.SubprocessError, OSError) as e:
            logger.warning(f"Notification failed: {e}")

        return False

    @staticmethod
    def say(text: str) -> bool:
        """Platform-adaptive TTS. Returns True on success."""
        caps = SubstrateDetector.detect()

        if not caps.has_say:
            logger.debug(f"TTS not available on {caps.platform}")
            return False

        try:
            subprocess.Popen(
                ["say", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(f"TTS failed: {e}")
            return False

    @staticmethod
    def trash_file(path: Path) -> bool:
        """Platform-adaptive file deletion (trash when possible).
        Returns True on success."""
        caps = SubstrateDetector.detect()

        if not path.exists():
            return False

        if caps.platform == DARWIN:
            try:
                subprocess.run(
                    [
                        "osascript", "-e",
                        f'tell app "Finder" to delete POSIX file "{path.resolve()}"',
                    ],
                    capture_output=True,
                    timeout=5,
                    check=True,
                )
                return True
            except (subprocess.SubprocessError, OSError):
                pass

        # Fallback: permanent delete
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return True
        except OSError as e:
            logger.warning(f"Delete failed: {e}")
            return False

    @staticmethod
    def ensure_config_dir() -> Path:
        """Ensure ~/.motherlabs exists. Returns the path."""
        caps = SubstrateDetector.detect()
        caps.config_dir.mkdir(parents=True, exist_ok=True)
        return caps.config_dir


# --- Internal helpers ---

def _find_spotlight(
    query: str,
    directory: Optional[Path],
    max_results: int,
    timeout: int,
) -> Optional[List[Path]]:
    """Search via macOS Spotlight. Returns None on failure."""
    cmd = ["mdfind"]
    if directory:
        cmd.extend(["-onlyin", str(directory)])
    cmd.append(query)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"mdfind timed out after {timeout}s")
        return None

    if result.returncode != 0:
        logger.warning(f"mdfind failed: {result.stderr.strip()}")
        return None

    lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
    paths = []
    for line in lines[:max_results]:
        p = Path(line)
        if p.exists():
            paths.append(p)
    return paths


def _find_glob(
    query: str,
    directory: Optional[Path],
    max_results: int,
) -> List[Path]:
    """Search via recursive glob. Cross-platform fallback."""
    search_root = directory or Path.home()
    pattern = f"**/*{query}*"

    results: List[Path] = []
    try:
        for match in search_root.glob(pattern):
            if match.is_file():
                results.append(match)
                if len(results) >= max_results:
                    break
    except (PermissionError, OSError) as e:
        logger.warning(f"Glob search error: {e}")

    return results
