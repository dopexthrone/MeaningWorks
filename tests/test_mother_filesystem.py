"""
Tests for Mother filesystem bridge — all mocked, no real filesystem operations.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from mother.filesystem import FileSystemBridge, _human_size, _relative_time


# --- Helpers ---

@pytest.fixture
def fs():
    """FileSystemBridge with home as allowed root."""
    return FileSystemBridge(allowed_roots=[Path.home()], file_access=True)


@pytest.fixture
def fs_disabled():
    """FileSystemBridge with file_access=False."""
    return FileSystemBridge(file_access=False)


@pytest.fixture
def fs_restricted(tmp_path):
    """FileSystemBridge restricted to tmp_path only."""
    return FileSystemBridge(allowed_roots=[tmp_path], file_access=True)


# --- TestHumanSize ---

class TestHumanSize:
    def test_bytes(self):
        assert _human_size(500) == "500 B"

    def test_kilobytes(self):
        result = _human_size(2048)
        assert "KB" in result

    def test_megabytes(self):
        result = _human_size(5 * 1024 * 1024)
        assert "MB" in result

    def test_zero(self):
        assert _human_size(0) == "0 B"


class TestRelativeTime:
    @patch("mother.filesystem.datetime")
    def test_just_now(self, mock_dt):
        mock_dt.now.return_value.timestamp.return_value = 1000.0
        assert _relative_time(999.5) == "just now"

    @patch("mother.filesystem.datetime")
    def test_minutes_ago(self, mock_dt):
        mock_dt.now.return_value.timestamp.return_value = 1000.0
        result = _relative_time(1000.0 - 300)  # 5 minutes ago
        assert "minute" in result

    @patch("mother.filesystem.datetime")
    def test_hours_ago(self, mock_dt):
        mock_dt.now.return_value.timestamp.return_value = 10000.0
        result = _relative_time(10000.0 - 7200)  # 2 hours ago
        assert "hour" in result

    @patch("mother.filesystem.datetime")
    def test_days_ago(self, mock_dt):
        mock_dt.now.return_value.timestamp.return_value = 200000.0
        result = _relative_time(200000.0 - 259200)  # 3 days ago
        assert "days ago" in result


# --- TestFileSystemSearch ---

class TestFileSystemSearch:
    @patch("subprocess.run")
    def test_mdfind_returns_results(self, mock_run, fs):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/Users/test/resume.pdf\n/Users/test/resume-v2.pdf\n",
        )
        with patch.object(fs, "_build_file_info") as mock_info:
            mock_info.side_effect = [
                {"path": "/Users/test/resume.pdf", "name": "resume.pdf", "size": 1000,
                 "size_human": "1000 B", "kind": "PDF", "modified": 0, "modified_human": "today"},
                {"path": "/Users/test/resume-v2.pdf", "name": "resume-v2.pdf", "size": 2000,
                 "size_human": "2.0 KB", "kind": "PDF", "modified": 0, "modified_human": "today"},
            ]
            with patch("mother.filesystem.sys") as mock_sys:
                mock_sys.platform = "darwin"
                results = fs.search("resume")
        assert len(results) == 2
        assert results[0]["name"] == "resume.pdf"

    @patch("subprocess.run")
    def test_mdfind_with_path_restriction(self, mock_run, fs):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        with patch("mother.filesystem.sys") as mock_sys:
            mock_sys.platform = "darwin"
            home = str(Path.home())
            fs.search("test", path=home)
        call_args = mock_run.call_args[0][0]
        assert "-onlyin" in call_args

    @patch("subprocess.run")
    def test_mdfind_timeout_returns_empty(self, mock_run, fs):
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired(cmd="mdfind", timeout=10)
        with patch("mother.filesystem.sys") as mock_sys:
            mock_sys.platform = "darwin"
            results = fs.search("test")
        assert results == []

    @patch("subprocess.run")
    def test_mdfind_respects_max_results(self, mock_run, fs):
        lines = "\n".join([f"/Users/test/file{i}.txt" for i in range(30)])
        mock_run.return_value = MagicMock(returncode=0, stdout=lines)
        with patch.object(fs, "_build_file_info") as mock_info:
            mock_info.return_value = {"path": "x", "name": "x", "size": 0,
                                       "size_human": "0 B", "kind": "file", "modified": 0,
                                       "modified_human": "now"}
            with patch("mother.filesystem.sys") as mock_sys:
                mock_sys.platform = "darwin"
                results = fs.search("file", max_results=5)
        assert len(results) == 5

    @patch("subprocess.run")
    def test_mdfind_failure_returns_empty(self, mock_run, fs):
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        with patch("mother.filesystem.sys") as mock_sys:
            mock_sys.platform = "darwin"
            results = fs.search("test")
        assert results == []

    def test_glob_fallback_on_non_darwin(self, tmp_path):
        # Create a real file in tmp_path for glob to find
        (tmp_path / "readme.txt").write_text("hello")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with patch("mother.filesystem.sys") as mock_sys:
            mock_sys.platform = "linux"
            results = fsb.search("readme", path=str(tmp_path))
        assert len(results) == 1
        assert results[0]["name"] == "readme.txt"

    def test_search_blocked_when_disabled(self, fs_disabled):
        with pytest.raises(PermissionError, match="disabled"):
            fs_disabled.search("test")

    def test_search_empty_results_parsed(self, fs):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="\n")
            with patch("mother.filesystem.sys") as mock_sys:
                mock_sys.platform = "darwin"
                results = fs.search("nonexistent_xyz")
        assert results == []


# --- TestFileSystemRead ---

class TestFileSystemRead:
    def test_read_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, Mother.")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        result = fsb.read_file(str(f))
        assert result["content"] == "Hello, Mother."
        assert result["binary"] is False
        assert result["truncated"] is False

    def test_read_respects_max_bytes(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 500)
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        result = fsb.read_file(str(f), max_bytes=100)
        assert len(result["content"]) <= 100
        assert result["truncated"] is True

    def test_read_binary_file(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(bytes(range(256)))
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        result = fsb.read_file(str(f))
        assert result["binary"] is True
        assert "[Binary file" in result["content"]

    def test_read_missing_file_raises(self, tmp_path):
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with pytest.raises(FileNotFoundError):
            fsb.read_file(str(tmp_path / "nope.txt"))

    def test_read_directory_raises(self, tmp_path):
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with pytest.raises(IsADirectoryError):
            fsb.read_file(str(tmp_path))


# --- TestFileSystemWrite ---

class TestFileSystemWrite:
    def test_write_creates_file(self, tmp_path):
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        target = tmp_path / "output.txt"
        result = fsb.write_file(str(target), "Hello")
        assert target.read_text() == "Hello"
        assert result["bytes_written"] == 5

    def test_write_creates_parent_dirs(self, tmp_path):
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        target = tmp_path / "sub" / "deep" / "file.txt"
        fsb.write_file(str(target), "nested")
        assert target.read_text() == "nested"

    def test_write_refuses_overwrite_by_default(self, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("original")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with pytest.raises(FileExistsError):
            fsb.write_file(str(f), "new content")

    def test_write_allows_overwrite_when_explicit(self, tmp_path):
        f = tmp_path / "exists.txt"
        f.write_text("original")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        fsb.write_file(str(f), "new content", overwrite=True)
        assert f.read_text() == "new content"


# --- TestFileSystemOps ---

class TestFileSystemOps:
    def test_move_file(self, tmp_path):
        src = tmp_path / "a.txt"
        src.write_text("data")
        dst = tmp_path / "b.txt"
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        result = fsb.move_file(str(src), str(dst))
        assert not src.exists()
        assert dst.read_text() == "data"
        assert "dst" in result

    def test_copy_file(self, tmp_path):
        src = tmp_path / "a.txt"
        src.write_text("data")
        dst = tmp_path / "copy.txt"
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        result = fsb.copy_file(str(src), str(dst))
        assert src.exists()
        assert dst.read_text() == "data"
        assert "dst" in result

    @patch("subprocess.run")
    def test_delete_uses_trash_on_macos(self, mock_run, tmp_path):
        f = tmp_path / "trash_me.txt"
        f.write_text("bye")
        mock_run.return_value = MagicMock(returncode=0)
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with patch("mother.filesystem.sys") as mock_sys:
            mock_sys.platform = "darwin"
            result = fsb.delete_file(str(f))
        assert result["method"] == "trash"
        assert "osascript" in mock_run.call_args[0][0][0]

    def test_delete_missing_file_raises(self, tmp_path):
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with pytest.raises(FileNotFoundError):
            fsb.delete_file(str(tmp_path / "nope.txt"))

    def test_list_dir(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "sub").mkdir()
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        results = fsb.list_dir(str(tmp_path))
        names = [r["name"] for r in results]
        assert "a.txt" in names
        assert "b.txt" in names
        assert "sub" in names

    def test_list_dir_with_pattern(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.py").write_text("b")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        results = fsb.list_dir(str(tmp_path), pattern="*.txt")
        assert len(results) == 1
        assert results[0]["name"] == "a.txt"

    def test_file_info(self, tmp_path):
        f = tmp_path / "info.txt"
        f.write_text("hello")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        info = fsb.file_info(str(f))
        assert info["name"] == "info.txt"
        assert info["size"] == 5
        assert info["kind"] == "text"

    def test_file_info_missing_raises(self, tmp_path):
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with pytest.raises(FileNotFoundError):
            fsb.file_info(str(tmp_path / "nope.txt"))

    def test_move_missing_source_raises(self, tmp_path):
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with pytest.raises(FileNotFoundError):
            fsb.move_file(str(tmp_path / "nope.txt"), str(tmp_path / "dst.txt"))

    def test_copy_missing_source_raises(self, tmp_path):
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with pytest.raises(FileNotFoundError):
            fsb.copy_file(str(tmp_path / "nope.txt"), str(tmp_path / "dst.txt"))


# --- TestFileSystemAccess ---

class TestFileSystemAccess:
    def test_blocked_when_file_access_false(self):
        fsb = FileSystemBridge(file_access=False)
        with pytest.raises(PermissionError, match="disabled"):
            fsb.read_file("/some/path")

    def test_path_outside_roots_rejected(self, tmp_path):
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with pytest.raises(PermissionError, match="outside"):
            fsb.read_file("/etc/passwd")

    def test_home_always_allowed_by_default(self):
        fsb = FileSystemBridge(file_access=True)
        # _check_access should not raise for home subpath
        home_sub = Path.home() / "test_placeholder"
        result = fsb._check_access(home_sub)
        assert result == home_sub.resolve()

    def test_write_blocked_when_disabled(self):
        fsb = FileSystemBridge(file_access=False)
        with pytest.raises(PermissionError, match="disabled"):
            fsb.write_file("/some/path", "content")

    def test_delete_blocked_when_disabled(self):
        fsb = FileSystemBridge(file_access=False)
        with pytest.raises(PermissionError, match="disabled"):
            fsb.delete_file("/some/path")


# --- TestKindDetection ---

class TestKindDetection:
    def test_pdf_detected(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        info = fsb.file_info(str(f))
        assert info["kind"] == "PDF"

    def test_python_detected(self, tmp_path):
        f = tmp_path / "script.py"
        f.write_text("print('hi')")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        info = fsb.file_info(str(f))
        assert info["kind"] == "Python"

    def test_unknown_extension(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("stuff")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        info = fsb.file_info(str(f))
        assert info["kind"] == "file"

    def test_directory_detected(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        info = fsb.file_info(str(d))
        assert info["kind"] == "directory"


# --- TestFileSystemEdit ---

class TestFileSystemEdit:
    def test_edit_replaces_text(self, tmp_path):
        f = tmp_path / "config.txt"
        f.write_text("host=localhost\nport=3000")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        result = fsb.edit_file(str(f), "localhost", "production.example.com")
        assert "production.example.com" in f.read_text()
        assert result["replacements"] == 1

    def test_edit_replaces_first_occurrence_only(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("foo bar foo baz foo")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        fsb.edit_file(str(f), "foo", "qux")
        assert f.read_text() == "qux bar foo baz foo"

    def test_edit_missing_text_raises(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with pytest.raises(ValueError, match="not found"):
            fsb.edit_file(str(f), "nonexistent", "replacement")

    def test_edit_missing_file_raises(self, tmp_path):
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with pytest.raises(FileNotFoundError):
            fsb.edit_file(str(tmp_path / "nope.txt"), "old", "new")

    def test_edit_directory_raises(self, tmp_path):
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with pytest.raises(IsADirectoryError):
            fsb.edit_file(str(tmp_path), "old", "new")

    def test_edit_preserves_rest_of_file(self, tmp_path):
        f = tmp_path / "data.txt"
        original = "line1\nline2\nline3\nline4"
        f.write_text(original)
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        fsb.edit_file(str(f), "line2", "CHANGED")
        assert f.read_text() == "line1\nCHANGED\nline3\nline4"

    def test_edit_blocked_when_disabled(self):
        fsb = FileSystemBridge(file_access=False)
        with pytest.raises(PermissionError, match="disabled"):
            fsb.edit_file("/some/path", "old", "new")

    def test_edit_multiline_replacement(self, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("start\nold line\nend")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        fsb.edit_file(str(f), "old line", "new line 1\nnew line 2")
        assert f.read_text() == "start\nnew line 1\nnew line 2\nend"


# --- TestFileSystemAppend ---

class TestFileSystemAppend:
    def test_append_to_existing(self, tmp_path):
        f = tmp_path / "log.txt"
        f.write_text("line1\n")
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        result = fsb.append_file(str(f), "line2\n")
        assert f.read_text() == "line1\nline2\n"
        assert result["bytes_appended"] == 6

    def test_append_creates_file_if_missing(self, tmp_path):
        f = tmp_path / "new.txt"
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        result = fsb.append_file(str(f), "first line")
        assert f.read_text() == "first line"
        assert result["bytes_appended"] == 10

    def test_append_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "sub" / "deep" / "log.txt"
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        fsb.append_file(str(f), "content")
        assert f.read_text() == "content"

    def test_append_to_directory_raises(self, tmp_path):
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        with pytest.raises(IsADirectoryError):
            fsb.append_file(str(tmp_path), "content")

    def test_append_blocked_when_disabled(self):
        fsb = FileSystemBridge(file_access=False)
        with pytest.raises(PermissionError, match="disabled"):
            fsb.append_file("/some/path", "content")

    def test_append_multiple_times(self, tmp_path):
        f = tmp_path / "accum.txt"
        fsb = FileSystemBridge(allowed_roots=[tmp_path], file_access=True)
        fsb.append_file(str(f), "a")
        fsb.append_file(str(f), "b")
        fsb.append_file(str(f), "c")
        assert f.read_text() == "abc"


# --- TestChatFileActionParsing ---

class TestChatFileActionParsing:
    """Test parse_response with file action markers."""

    def test_write_action_parsed(self):
        from mother.screens.chat import parse_response
        raw = "[ACTION:file]write: ~/test.txt | hello world[/ACTION][VOICE]Saved.[/VOICE]"
        parsed = parse_response(raw)
        assert parsed["action"] == "file"
        assert "write:" in parsed["action_arg"]
        assert "hello world" in parsed["action_arg"]

    def test_edit_action_parsed(self):
        from mother.screens.chat import parse_response
        raw = "[ACTION:file]edit: ~/config.json | old_value -> new_value[/ACTION]"
        parsed = parse_response(raw)
        assert parsed["action"] == "file"
        assert "edit:" in parsed["action_arg"]
        assert "old_value -> new_value" in parsed["action_arg"]

    def test_append_action_parsed(self):
        from mother.screens.chat import parse_response
        raw = "[ACTION:file]append: ~/notes.txt | new note[/ACTION]"
        parsed = parse_response(raw)
        assert parsed["action"] == "file"
        assert "append:" in parsed["action_arg"]

    def test_create_action_parsed(self):
        from mother.screens.chat import parse_response
        raw = "[ACTION:file]create: ~/new.txt | content here[/ACTION]"
        parsed = parse_response(raw)
        assert parsed["action"] == "file"
        assert "create:" in parsed["action_arg"]
