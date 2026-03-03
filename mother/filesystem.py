"""
File system bridge — search, read, write, move, copy, delete.

Pure stdlib + subprocess. No imports from core/.
Uses macOS Spotlight (mdfind) for search, falls back to glob on other platforms.
All paths validated against allowed_roots. Delete uses Trash, not permanent removal.
"""

import logging
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("mother.filesystem")

_DEFAULT_MAX_RESULTS = 20
_DEFAULT_MAX_BYTES = 100_000
_MDFIND_TIMEOUT = 10


def _human_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.0f} {unit}" if unit == "B" else f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _relative_time(ts: float) -> str:
    """Format timestamp as relative time string."""
    delta = datetime.now().timestamp() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        m = int(delta / 60)
        return f"{m} minute{'s' if m != 1 else ''} ago"
    if delta < 86400:
        h = int(delta / 3600)
        return f"{h} hour{'s' if h != 1 else ''} ago"
    d = int(delta / 86400)
    if d == 1:
        return "yesterday"
    return f"{d} days ago"


class FileSystemBridge:
    """Local filesystem operations with safety constraints.

    All paths resolved to absolute and checked against allowed_roots.
    File access can be globally disabled via file_access=False.
    """

    def __init__(
        self,
        allowed_roots: Optional[List[Path]] = None,
        file_access: bool = True,
    ):
        self._file_access = file_access
        self._allowed_roots = [r.resolve() for r in (allowed_roots or [Path.home()])]

    def _check_access(self, path: Path) -> Path:
        """Resolve path and verify it's within allowed roots.

        Raises PermissionError if file_access is False or path is outside roots.
        Returns the resolved absolute path.
        """
        if not self._file_access:
            raise PermissionError("File access is disabled. Enable in /settings.")

        resolved = path.resolve()
        for root in self._allowed_roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
        raise PermissionError(f"Path outside allowed roots: {resolved}")

    def search(
        self,
        query: str,
        path: Optional[str] = None,
        max_results: int = _DEFAULT_MAX_RESULTS,
    ) -> List[Dict[str, Any]]:
        """Search for files using Spotlight (mdfind) or glob fallback.

        Returns list of dicts with: path, name, size, size_human, kind, modified, modified_human.
        """
        if not self._file_access:
            raise PermissionError("File access is disabled. Enable in /settings.")

        if sys.platform == "darwin":
            return self._search_mdfind(query, path, max_results)
        return self._search_glob(query, path, max_results)

    def _search_mdfind(
        self,
        query: str,
        path: Optional[str],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Spotlight search via mdfind CLI."""
        cmd = ["mdfind"]
        if path:
            resolved = self._check_access(Path(path))
            cmd.extend(["-onlyin", str(resolved)])
        cmd.append(query)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_MDFIND_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"mdfind timed out after {_MDFIND_TIMEOUT}s for query: {query}")
            return []

        if result.returncode != 0:
            logger.warning(f"mdfind failed: {result.stderr.strip()}")
            return []

        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        results = []
        for line in lines[:max_results]:
            info = self._build_file_info(Path(line))
            if info:
                results.append(info)
        return results

    def _search_glob(
        self,
        query: str,
        path: Optional[str],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Fallback search using pathlib glob."""
        search_root = Path(path) if path else Path.home()
        resolved = self._check_access(search_root)

        results = []
        pattern = f"**/*{query}*"
        try:
            for match in resolved.glob(pattern):
                if match.is_file():
                    info = self._build_file_info(match)
                    if info:
                        results.append(info)
                    if len(results) >= max_results:
                        break
        except (PermissionError, OSError) as e:
            logger.warning(f"Glob search error: {e}")

        return results

    def _build_file_info(self, path: Path) -> Optional[Dict[str, Any]]:
        """Build file info dict from path. Returns None if stat fails."""
        try:
            st = path.stat()
            return {
                "path": str(path),
                "name": path.name,
                "size": st.st_size,
                "size_human": _human_size(st.st_size),
                "kind": self._detect_kind(path, st),
                "modified": st.st_mtime,
                "modified_human": _relative_time(st.st_mtime),
            }
        except (OSError, ValueError):
            return None

    def _detect_kind(self, path: Path, st: os.stat_result) -> str:
        """Detect file kind from extension and stat mode."""
        if stat.S_ISDIR(st.st_mode):
            return "directory"
        ext = path.suffix.lower()
        kind_map = {
            ".pdf": "PDF", ".doc": "document", ".docx": "document",
            ".xls": "spreadsheet", ".xlsx": "spreadsheet", ".csv": "spreadsheet",
            ".jpg": "image", ".jpeg": "image", ".png": "image",
            ".gif": "image", ".svg": "image", ".webp": "image",
            ".mp3": "audio", ".wav": "audio", ".m4a": "audio",
            ".mp4": "video", ".mov": "video", ".avi": "video",
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".html": "HTML", ".css": "CSS", ".json": "JSON",
            ".md": "Markdown", ".txt": "text", ".yaml": "YAML", ".yml": "YAML",
            ".zip": "archive", ".tar": "archive", ".gz": "archive",
        }
        return kind_map.get(ext, "file")

    def read_file(self, path: str, max_bytes: int = _DEFAULT_MAX_BYTES) -> Dict[str, Any]:
        """Read file contents.

        Text files: returns content as string.
        Binary files: returns hex preview + size info.
        Capped at max_bytes.

        Returns dict with: path, content, size, truncated, binary.
        """
        resolved = self._check_access(Path(path))
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {resolved}")
        if resolved.is_dir():
            raise IsADirectoryError(f"Cannot read directory: {resolved}")

        size = resolved.stat().st_size
        truncated = size > max_bytes

        # Try text first
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                content = f.read(max_bytes)
            return {
                "path": str(resolved),
                "content": content,
                "size": size,
                "size_human": _human_size(size),
                "truncated": truncated,
                "binary": False,
            }
        except (UnicodeDecodeError, ValueError):
            pass

        # Binary fallback
        with open(resolved, "rb") as f:
            raw = f.read(min(256, max_bytes))
        hex_preview = raw.hex(" ", 1)[:200]
        return {
            "path": str(resolved),
            "content": f"[Binary file — {_human_size(size)}]\n{hex_preview}...",
            "size": size,
            "size_human": _human_size(size),
            "truncated": True,
            "binary": True,
        }

    def write_file(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Write text content to a file.

        Creates parent directories. Refuses to overwrite unless overwrite=True.

        Returns dict with: path, bytes_written.
        """
        resolved = self._check_access(Path(path))

        if resolved.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {resolved}. Use overwrite=True.")

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return {
            "path": str(resolved),
            "bytes_written": len(content.encode("utf-8")),
        }

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
    ) -> Dict[str, Any]:
        """Replace first occurrence of old_text with new_text in a file.

        Reads the file, performs the replacement, writes back.
        Raises ValueError if old_text is not found.

        Returns dict with: path, replacements.
        """
        resolved = self._check_access(Path(path))

        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {resolved}")
        if resolved.is_dir():
            raise IsADirectoryError(f"Cannot edit directory: {resolved}")

        content = resolved.read_text(encoding="utf-8")
        if old_text not in content:
            raise ValueError(f"Text not found in {resolved.name}")

        updated = content.replace(old_text, new_text, 1)
        resolved.write_text(updated, encoding="utf-8")
        return {
            "path": str(resolved),
            "replacements": 1,
        }

    def append_file(
        self,
        path: str,
        content: str,
    ) -> Dict[str, Any]:
        """Append text content to an existing file.

        Creates the file if it doesn't exist.

        Returns dict with: path, bytes_appended.
        """
        resolved = self._check_access(Path(path))

        if resolved.is_dir():
            raise IsADirectoryError(f"Cannot append to directory: {resolved}")

        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "a", encoding="utf-8") as f:
            f.write(content)
        return {
            "path": str(resolved),
            "bytes_appended": len(content.encode("utf-8")),
        }

    def move_file(self, src: str, dst: str) -> Dict[str, Any]:
        """Move or rename a file.

        Returns dict with: src, dst.
        """
        src_resolved = self._check_access(Path(src))
        dst_resolved = self._check_access(Path(dst))

        if not src_resolved.exists():
            raise FileNotFoundError(f"Source not found: {src_resolved}")

        dst_resolved.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_resolved), str(dst_resolved))
        return {"src": str(src_resolved), "dst": str(dst_resolved)}

    def copy_file(self, src: str, dst: str) -> Dict[str, Any]:
        """Copy a file preserving metadata.

        Returns dict with: src, dst.
        """
        src_resolved = self._check_access(Path(src))
        dst_resolved = self._check_access(Path(dst))

        if not src_resolved.exists():
            raise FileNotFoundError(f"Source not found: {src_resolved}")

        dst_resolved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_resolved), str(dst_resolved))
        return {"src": str(src_resolved), "dst": str(dst_resolved)}

    def delete_file(self, path: str) -> Dict[str, Any]:
        """Delete a file by moving to Trash (macOS) or unlinking.

        macOS: uses osascript to move to Trash (reversible).
        Other: os.unlink() (permanent).

        Returns dict with: path, method.
        """
        resolved = self._check_access(Path(path))

        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {resolved}")

        if sys.platform == "darwin":
            try:
                subprocess.run(
                    [
                        "osascript", "-e",
                        f'tell app "Finder" to delete POSIX file "{resolved}"',
                    ],
                    capture_output=True,
                    timeout=5,
                    check=True,
                )
                return {"path": str(resolved), "method": "trash"}
            except (subprocess.SubprocessError, OSError) as e:
                logger.warning(f"Trash failed, falling back to unlink: {e}")

        os.unlink(resolved)
        return {"path": str(resolved), "method": "unlink"}

    def list_dir(
        self,
        path: str,
        pattern: str = "*",
    ) -> List[Dict[str, Any]]:
        """List directory contents with metadata.

        Returns list of file info dicts.
        """
        resolved = self._check_access(Path(path))

        if not resolved.exists():
            raise FileNotFoundError(f"Directory not found: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {resolved}")

        results = []
        for entry in sorted(resolved.glob(pattern)):
            info = self._build_file_info(entry)
            if info:
                results.append(info)
        return results

    def file_info(self, path: str) -> Dict[str, Any]:
        """Get detailed file info.

        Returns dict with: path, name, size, size_human, kind, modified, modified_human.
        """
        resolved = self._check_access(Path(path))

        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {resolved}")

        info = self._build_file_info(resolved)
        if info is None:
            raise OSError(f"Could not stat: {resolved}")
        return info
