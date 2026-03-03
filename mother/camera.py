"""
Webcam capture bridge — OpenCV to base64 JPEG.

LEAF module. Pure stdlib + cv2. No subprocess. Cross-platform.
All errors caught — capture failure returns None, never raises.
Auto-detects working device index, caches it. Resizes large frames.
"""

import asyncio
import base64
import logging
import sys
from typing import Optional

logger = logging.getLogger("mother.camera")


def is_camera_available() -> bool:
    """True if OpenCV available and at least one camera device opens (indices 0-9)."""
    try:
        import cv2
        plat = sys.platform
        backend = cv2.CAP_ANY
        if plat == 'darwin' and hasattr(cv2, 'CAP_AVFOUNDATION'):
            backend = cv2.CAP_AVFOUNDATION
        elif plat == 'win32' and hasattr(cv2, 'CAP_DSHOW'):
            backend = cv2.CAP_DSHOW
        elif plat.startswith('linux') and hasattr(cv2, 'CAP_V4L2'):
            backend = cv2.CAP_V4L2
        for i in range(10):
            cap = cv2.VideoCapture(i, backend)
            try:
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        return True
            finally:
                cap.release()
        return False
    except Exception:
        return False


class CameraBridge:
    """Async webcam capture via OpenCV.

    Captures a single frame from the webcam as JPEG, returns base64-encoded string.
    All exceptions caught — capture failure returns None, never fatal.
    Auto-detects and caches working device index (tries 0-4).
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._cv2 = None
        self._working_device: Optional[int] = None

    @property
    def enabled(self) -> bool:
        """True if enabled flag and OpenCV import succeeds."""
        if not self._enabled:
            return False
        if self._cv2 is None:
            try:
                import cv2
                self._cv2 = cv2
            except ImportError:
                return False
        return True

    def _capture_sync(self, device_index: Optional[int] = None) -> Optional[str]:
        """Capture a single webcam frame and return base64-encoded JPEG.

        device_index: override cached device (default: auto-detect then cache).

        Returns base64 string or None on failure.
        """
        if not self.enabled:
            return None

        cv2 = self._cv2

        # Platform-specific backend selection
        plat = sys.platform
        backend = cv2.CAP_ANY
        if plat == 'darwin' and hasattr(cv2, 'CAP_AVFOUNDATION'):
            backend = cv2.CAP_AVFOUNDATION
        elif plat == 'win32' and hasattr(cv2, 'CAP_DSHOW'):
            backend = cv2.CAP_DSHOW
        elif plat.startswith('linux') and hasattr(cv2, 'CAP_V4L2'):
            backend = cv2.CAP_V4L2

        use_index = device_index
        if use_index is None:
            use_index = self._working_device
        if use_index is None:
            logger.info("Auto-detecting working camera device...")
            for test_index in range(10):  # Try indices 0-9
                cap = cv2.VideoCapture(test_index, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        self._working_device = test_index
                        logger.info(f"Found working camera at index {test_index}")
                        cap.release()
                        use_index = test_index
                        break
                    cap.release()
            if use_index is None:
                logger.warning("No working camera device found (0-4)")
                return None

        cap = cv2.VideoCapture(use_index, backend)
        try:
            if not cap.isOpened():
                logger.warning(f"Camera device {use_index} failed to open")
                return None

            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                logger.warning(f"Failed to grab valid frame from camera {use_index}")
                return None

            # Resize if too large (for speed/cost)
            h, w = frame.shape[:2]
            max_w, max_h = 640, 480
            if w > max_w or h > max_h:
                scale_w = max_w / w
                scale_h = max_h / h
                scale = min(scale_w, scale_h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # JPEG encode at 85% quality
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            ret_encode, buffer = cv2.imencode('.jpg', frame, encode_params)
            if not ret_encode:
                logger.warning("Failed to encode frame to JPEG")
                return None

            return base64.b64encode(buffer).decode('ascii')

        except Exception as e:
            logger.warning(f"Camera capture error (non-fatal): {e}")
            return None
        finally:
            cap.release()

    async def capture_frame(self, device_index: Optional[int] = None) -> Optional[str]:
        """Async wrapper: capture in threadpool."""
        if not self.enabled:
            return None
        return await asyncio.to_thread(self._capture_sync, device_index)
