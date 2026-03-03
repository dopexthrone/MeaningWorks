"""
Tests for Mother webcam camera bridge (OpenCV / cv2 implementation).

All tests mocked. No real camera captures. No real cv2 import. No numpy.
"""

import asyncio
import base64
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers — build mock cv2 and mock frames without numpy
# ---------------------------------------------------------------------------

def _make_frame(shape=(480, 640, 3), size=None):
    """Build a mock frame with .shape and .size like a numpy array."""
    frame = MagicMock()
    frame.shape = shape
    frame.size = size if size is not None else (shape[0] * shape[1] * shape[2])
    return frame


def _make_mock_cv2(opened=True, read_ok=True, frame_shape=(480, 640, 3),
                   encode_ok=True, encoded_bytes=b"\xff\xd8JPEG"):
    """Build a fully wired mock cv2 module.

    Returns (mock_cv2, mock_cap) so callers can further customize.

    Uses spec=[] to prevent MagicMock auto-creating attributes for hasattr
    checks. Platform-specific backend constants (CAP_AVFOUNDATION, CAP_DSHOW,
    CAP_V4L2) are intentionally absent so all test paths use CAP_ANY.
    """
    mock_cv2 = MagicMock(spec=[])
    mock_cap = MagicMock()

    mock_cv2.VideoCapture = MagicMock(return_value=mock_cap)
    mock_cap.isOpened.return_value = opened

    if frame_shape is not None:
        frame = _make_frame(frame_shape)
    else:
        frame = None

    mock_cap.read.return_value = (read_ok, frame)

    # imencode returns (bool, buffer-like with tobytes support)
    buf = MagicMock()
    buf.__bytes__ = lambda self: encoded_bytes
    # base64.b64encode needs a bytes-like; use a real bytes wrapper
    mock_cv2.imencode = MagicMock(return_value=(encode_ok, encoded_bytes))

    # Constants needed by the implementation
    mock_cv2.CAP_ANY = 0
    mock_cv2.IMWRITE_JPEG_QUALITY = 1
    mock_cv2.INTER_AREA = 3
    mock_cv2.resize = MagicMock()

    return mock_cv2, mock_cap


# ---------------------------------------------------------------------------
# Test is_camera_available()
# ---------------------------------------------------------------------------

class TestCameraAvailability:
    """Test OpenCV-based camera detection."""

    def test_available_when_cv2_and_device_opens(self):
        mock_cv2, mock_cap = _make_mock_cv2(opened=True)
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            import importlib
            import mother.camera as cam
            importlib.reload(cam)
            assert cam.is_camera_available() is True
            # release called via try/finally — at least once (first success)
            assert mock_cap.release.called

    def test_not_available_when_device_does_not_open(self):
        mock_cv2, mock_cap = _make_mock_cv2(opened=False)
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            import importlib
            import mother.camera as cam
            importlib.reload(cam)
            assert cam.is_camera_available() is False
            # release called via try/finally for each of 10 device indices
            assert mock_cap.release.call_count == 10

    def test_not_available_when_cv2_import_fails(self):
        """If cv2 is not installed, is_camera_available returns False."""
        with patch.dict("sys.modules", {"cv2": None}):
            import importlib
            import mother.camera as cam
            importlib.reload(cam)
            assert cam.is_camera_available() is False

    def test_returns_bool_type(self):
        mock_cv2, _ = _make_mock_cv2(opened=True)
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            import importlib
            import mother.camera as cam
            importlib.reload(cam)
            result = cam.is_camera_available()
            assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test CameraBridge creation
# ---------------------------------------------------------------------------

class TestCameraBridgeCreation:
    """Test CameraBridge instantiation."""

    def test_create_default(self):
        from mother.camera import CameraBridge
        bridge = CameraBridge()
        assert bridge._enabled is True
        assert bridge._cv2 is None
        assert bridge._working_device is None

    def test_create_disabled(self):
        from mother.camera import CameraBridge
        bridge = CameraBridge(enabled=False)
        assert bridge._enabled is False

    def test_create_with_explicit_enabled(self):
        from mother.camera import CameraBridge
        bridge = CameraBridge(enabled=True)
        assert bridge._enabled is True


# ---------------------------------------------------------------------------
# Test enabled property
# ---------------------------------------------------------------------------

class TestCameraBridgeEnabled:
    """Test the enabled property (flag + cv2 import gate)."""

    def test_enabled_when_flag_true_and_cv2_available(self):
        from mother.camera import CameraBridge
        bridge = CameraBridge(enabled=True)
        mock_cv2 = MagicMock()
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            assert bridge.enabled is True
            assert bridge._cv2 is mock_cv2

    def test_disabled_when_flag_false(self):
        from mother.camera import CameraBridge
        bridge = CameraBridge(enabled=False)
        assert bridge.enabled is False
        assert bridge._cv2 is None

    def test_disabled_when_cv2_import_fails(self):
        from mother.camera import CameraBridge
        bridge = CameraBridge(enabled=True)
        with patch.dict("sys.modules", {"cv2": None}):
            assert bridge.enabled is False

    def test_cv2_cached_after_first_access(self):
        from mother.camera import CameraBridge
        bridge = CameraBridge(enabled=True)
        mock_cv2 = MagicMock()
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            _ = bridge.enabled
            _ = bridge.enabled
            assert bridge._cv2 is mock_cv2


# ---------------------------------------------------------------------------
# Test _capture_sync — success path
# ---------------------------------------------------------------------------

class TestCaptureSyncSuccess:
    """Test successful synchronous capture."""

    def test_capture_returns_base64_string(self):
        from mother.camera import CameraBridge
        mock_cv2, mock_cap = _make_mock_cv2()
        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 0

        result = bridge._capture_sync()

        assert result is not None
        assert isinstance(result, str)
        # Verify it is valid base64 that decodes to original bytes
        decoded = base64.b64decode(result)
        assert decoded == b"\xff\xd8JPEG"

    def test_capture_with_explicit_device_index(self):
        from mother.camera import CameraBridge
        mock_cv2, mock_cap = _make_mock_cv2()
        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 0

        result = bridge._capture_sync(device_index=2)

        assert result is not None
        # VideoCapture called with (device_index, backend)
        mock_cv2.VideoCapture.assert_called_with(2, 0)

    def test_capture_releases_device(self):
        from mother.camera import CameraBridge
        mock_cv2, mock_cap = _make_mock_cv2()
        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 0

        bridge._capture_sync()

        mock_cap.release.assert_called()

    def test_capture_encodes_jpeg_at_quality_85(self):
        from mother.camera import CameraBridge
        mock_cv2, _ = _make_mock_cv2()
        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 0

        bridge._capture_sync()

        mock_cv2.imencode.assert_called_once()
        args = mock_cv2.imencode.call_args
        assert args[0][0] == '.jpg'
        encode_params = args[0][2]
        assert encode_params[1] == 85


# ---------------------------------------------------------------------------
# Test _capture_sync — auto-detection
# ---------------------------------------------------------------------------

class TestCaptureSyncAutoDetect:
    """Test device auto-detection (indices 0-4)."""

    def test_auto_detects_first_working_device(self):
        from mother.camera import CameraBridge
        mock_cv2 = MagicMock(spec=[])
        mock_cv2.CAP_ANY = 0
        mock_cv2.IMWRITE_JPEG_QUALITY = 1
        mock_cv2.INTER_AREA = 3

        frame = _make_frame((480, 640, 3))
        mock_cv2.imencode = MagicMock(return_value=(True, b"JPEG"))

        # First device fails to open, second opens but read fails, third works
        cap_fail = MagicMock()
        cap_fail.isOpened.return_value = False

        cap_fail2 = MagicMock()
        cap_fail2.isOpened.return_value = True
        cap_fail2.read.return_value = (False, None)

        cap_ok_detect = MagicMock()
        cap_ok_detect.isOpened.return_value = True
        cap_ok_detect.read.return_value = (True, frame)

        cap_ok_use = MagicMock()
        cap_ok_use.isOpened.return_value = True
        cap_ok_use.read.return_value = (True, frame)

        mock_cv2.VideoCapture = MagicMock(side_effect=[cap_fail, cap_fail2, cap_ok_detect, cap_ok_use])

        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2

        result = bridge._capture_sync()

        assert result is not None
        assert bridge._working_device == 2

    def test_no_working_device_returns_none(self):
        from mother.camera import CameraBridge
        mock_cv2 = MagicMock(spec=[])
        mock_cv2.CAP_ANY = 0
        mock_cv2.IMWRITE_JPEG_QUALITY = 1
        mock_cv2.INTER_AREA = 3

        cap_fail = MagicMock()
        cap_fail.isOpened.return_value = False
        mock_cv2.VideoCapture = MagicMock(return_value=cap_fail)

        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2

        result = bridge._capture_sync()

        assert result is None
        assert bridge._working_device is None

    def test_cached_device_skips_auto_detect(self):
        from mother.camera import CameraBridge
        mock_cv2, mock_cap = _make_mock_cv2()
        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 3

        bridge._capture_sync()

        # VideoCapture called with (cached_device, backend=CAP_ANY=0)
        mock_cv2.VideoCapture.assert_called_with(3, 0)


# ---------------------------------------------------------------------------
# Test _capture_sync — resize logic
# ---------------------------------------------------------------------------

class TestCaptureSyncResize:
    """Test frame resize for oversized captures."""

    def test_large_frame_gets_resized(self):
        from mother.camera import CameraBridge
        mock_cv2, mock_cap = _make_mock_cv2(frame_shape=(1080, 1920, 3))
        resized_frame = _make_frame((480, 640, 3))
        mock_cv2.resize.return_value = resized_frame

        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 0

        bridge._capture_sync()

        mock_cv2.resize.assert_called_once()
        call_args = mock_cv2.resize.call_args
        new_w, new_h = call_args[0][1]
        assert new_w <= 640
        assert new_h <= 480

    def test_small_frame_not_resized(self):
        from mother.camera import CameraBridge
        mock_cv2, mock_cap = _make_mock_cv2(frame_shape=(240, 320, 3))

        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 0

        bridge._capture_sync()

        mock_cv2.resize.assert_not_called()

    def test_resize_preserves_aspect_ratio(self):
        from mother.camera import CameraBridge
        mock_cv2, mock_cap = _make_mock_cv2(frame_shape=(1000, 2000, 3))
        mock_cv2.resize.return_value = _make_frame((320, 640, 3))

        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 0

        bridge._capture_sync()

        call_args = mock_cv2.resize.call_args
        new_w, new_h = call_args[0][1]
        # scale = min(640/2000, 480/1000) = min(0.32, 0.48) = 0.32
        # new_w = int(2000*0.32) = 640, new_h = int(1000*0.32) = 320
        assert new_w == 640
        assert new_h == 320


# ---------------------------------------------------------------------------
# Test _capture_sync — failure paths
# ---------------------------------------------------------------------------

class TestCaptureSyncFailures:
    """Test all failure modes return None without raising."""

    def test_disabled_flag_returns_none(self):
        from mother.camera import CameraBridge
        bridge = CameraBridge(enabled=False)
        assert bridge._capture_sync() is None

    def test_device_fails_to_open_returns_none(self):
        from mother.camera import CameraBridge
        mock_cv2, mock_cap = _make_mock_cv2(opened=False)
        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 0

        assert bridge._capture_sync() is None

    def test_read_fails_returns_none(self):
        from mother.camera import CameraBridge
        mock_cv2, mock_cap = _make_mock_cv2(read_ok=False)
        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 0

        assert bridge._capture_sync() is None

    def test_empty_frame_returns_none(self):
        from mother.camera import CameraBridge
        mock_cv2, mock_cap = _make_mock_cv2()
        empty_frame = _make_frame((0, 0, 0), size=0)
        mock_cap.read.return_value = (True, empty_frame)

        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 0

        assert bridge._capture_sync() is None

    def test_none_frame_returns_none(self):
        from mother.camera import CameraBridge
        mock_cv2, mock_cap = _make_mock_cv2()
        mock_cap.read.return_value = (True, None)

        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 0

        assert bridge._capture_sync() is None

    def test_imencode_failure_returns_none(self):
        from mother.camera import CameraBridge
        mock_cv2, mock_cap = _make_mock_cv2(encode_ok=False)
        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 0

        assert bridge._capture_sync() is None

    def test_exception_during_capture_returns_none(self):
        from mother.camera import CameraBridge
        mock_cv2, mock_cap = _make_mock_cv2()
        mock_cap.read.side_effect = RuntimeError("hardware fault")

        bridge = CameraBridge(enabled=True)
        bridge._cv2 = mock_cv2
        bridge._working_device = 0

        result = bridge._capture_sync()
        assert result is None
        mock_cap.release.assert_called()


# ---------------------------------------------------------------------------
# Test async capture_frame
# ---------------------------------------------------------------------------

class TestCameraAsync:
    """Test the async capture_frame wrapper."""

    def test_async_capture_delegates_to_sync(self):
        from mother.camera import CameraBridge
        bridge = CameraBridge(enabled=True)
        mock_cv2 = MagicMock()
        bridge._cv2 = mock_cv2

        async def _run():
            with patch.object(bridge, "_capture_sync", return_value="b64data") as mock_sync:
                result = await bridge.capture_frame()
                assert result == "b64data"
                mock_sync.assert_called_once_with(None)

        asyncio.run(_run())

    def test_async_capture_passes_device_index(self):
        from mother.camera import CameraBridge
        bridge = CameraBridge(enabled=True)
        mock_cv2 = MagicMock()
        bridge._cv2 = mock_cv2

        async def _run():
            with patch.object(bridge, "_capture_sync", return_value="data") as mock_sync:
                await bridge.capture_frame(device_index=2)
                mock_sync.assert_called_once_with(2)

        asyncio.run(_run())

    def test_async_capture_disabled_returns_none(self):
        from mother.camera import CameraBridge
        bridge = CameraBridge(enabled=False)

        async def _run():
            result = await bridge.capture_frame()
            assert result is None

        asyncio.run(_run())

    def test_async_capture_no_cv2_returns_none(self):
        from mother.camera import CameraBridge
        bridge = CameraBridge(enabled=True)
        with patch.dict("sys.modules", {"cv2": None}):
            async def _run():
                result = await bridge.capture_frame()
                assert result is None

            asyncio.run(_run())
