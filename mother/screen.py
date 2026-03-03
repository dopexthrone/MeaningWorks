"""
Screen capture bridge — macOS screencapture to base64 PNG.

LEAF module. Pure stdlib + subprocess. No imports from core/.
Follows the bridge pattern established by voice.py and filesystem.py.
All errors caught — capture failure returns None, never raises.
"""

import asyncio
import base64
import logging
import os
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

logger = logging.getLogger("mother.screen")


def is_screen_capture_available() -> bool:
    """True if running on macOS (screencapture CLI available)."""
    return sys.platform == "darwin"


class ScreenCaptureBridge:
    """Async screen capture via macOS screencapture CLI.

    Captures the screen as a PNG, returns base64-encoded string.
    All exceptions caught — capture failure returns None, never fatal.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Screen capture requires both the flag AND macOS."""
        return self._enabled and is_screen_capture_available()

    def _capture_sync(
        self,
        display: int = 1,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[str]:
        """Capture screen and return base64-encoded PNG.

        display: display number (1 = main)
        region: optional (x, y, width, height) crop rectangle

        Returns base64 string or None on failure.
        """
        if not self.enabled:
            return None

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="mother_capture_")
            os.close(fd)

            cmd = ["screencapture", "-x", "-t", "png"]

            # Display selection
            if display != 1:
                cmd.extend(["-D", str(display)])

            # Region capture
            if region is not None:
                x, y, w, h = region
                cmd.extend(["-R", f"{x},{y},{w},{h}"])

            cmd.append(tmp_path)

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.warning(f"screencapture failed: return code {result.returncode}")
                return None

            # Read and encode
            with open(tmp_path, "rb") as f:
                data = f.read()

            if not data:
                logger.warning("screencapture produced empty file")
                return None

            return base64.b64encode(data).decode("ascii")

        except subprocess.TimeoutExpired:
            logger.warning("screencapture timed out")
            return None
        except Exception as e:
            logger.warning(f"Screen capture error (non-fatal): {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    async def capture_screen(
        self,
        display: int = 1,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[str]:
        """Async screen capture. Returns base64 PNG or None.

        Runs in thread to avoid blocking the event loop.
        """
        if not self.enabled:
            return None

        return await asyncio.to_thread(self._capture_sync, display, region)
