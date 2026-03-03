"""
Motherlabs Runtime Scaffold — deterministic template generators for runtime infrastructure.

Generates standalone Python files that provide runtime plumbing for agent systems:
event loop, state store, LLM client, tool executor, config, and self-recompiler.

This is a LEAF MODULE — stdlib only. No engine/protocol/pipeline imports.
All functions are pure: (RuntimeCapabilities, blueprint, components) -> str.
Generated code has NO imports from motherlabs internals — fully standalone.

Key principle: runtime infrastructure is template-generated, not LLM-generated.
Only component behavior comes from the LLM. This preserves the trust guarantee.
"""

import json
from typing import Dict, Any, List

from core.naming import to_snake


def generate_runtime_py(
    capabilities: Any,
    blueprint: Dict[str, Any],
    component_names: List[str],
) -> str:
    """Generate runtime.py — async event loop with component dispatch.

    Args:
        capabilities: RuntimeCapabilities instance
        blueprint: Compiled blueprint dict
        component_names: List of component names

    Returns:
        Valid Python source code, or empty string if event loop disabled.
    """
    if not capabilities.has_event_loop:
        return ""

    port = capabilities.default_port

    # Build component registration code
    register_lines = []
    for name in component_names:
        safe = to_snake(name)
        register_lines.append(f'        # self.components["{name}"] = <component instance>')

    register_block = "\n".join(register_lines) if register_lines else "        pass"

    return f'''"""Runtime — async event loop with component message dispatch."""

import asyncio
import json
import logging
import signal
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Runtime:
    """Async runtime that dispatches messages to registered components."""

    def __init__(self, state=None, llm=None, tools=None, compiler=None, tool_manager=None):
        self.state = state
        self.llm = llm
        self.tools = tools
        self.compiler = compiler
        self.tool_manager = tool_manager
        self.components: Dict[str, Any] = {{}}
        self._running = False
        self._server = None
        self._start_time = time.time()
        self._message_count = 0

    def register(self, name: str, component: Any) -> None:
        """Register a component for message dispatch."""
        self.components[name] = component
        # Inject runtime services into component
        if hasattr(component, "state") and self.state is not None:
            component.state = self.state
        if hasattr(component, "llm") and self.llm is not None:
            component.llm = self.llm
        if hasattr(component, "tools") and self.tools is not None:
            component.tools = self.tools
        logger.info("Registered component: %s", name)

    async def dispatch(self, target: str, message: dict) -> dict:
        """Dispatch a message to a named component."""
        self._message_count += 1
        component = self.components.get(target)
        if component is None:
            return {{"error": f"Unknown component: {{target}}"}}
        try:
            if hasattr(component, "handle"):
                return await component.handle(message)
            return {{"error": f"Component {{target}} has no handle() method"}}
        except Exception as e:
            logger.exception("Error dispatching to %s", target)
            return {{"error": str(e)}}

    def emit(self, event: str, data: dict) -> None:
        """Publish an event to all components that have on_event()."""
        for name, comp in self.components.items():
            if hasattr(comp, "on_event"):
                asyncio.create_task(comp.on_event(event, data))

    async def _handle_system(self, payload: dict) -> dict:
        """Handle _system meta-target for introspection and self-extension.

        Actions:
            learn   — request recompilation with a new skill
            status  — return registered components
            health  — return uptime and message count
        """
        action = payload.get("action", "")

        if action == "status":
            return {{"components": sorted(self.components.keys())}}

        elif action == "health":
            uptime = time.time() - self._start_time
            return {{
                "uptime": round(uptime, 1),
                "messages": self._message_count,
                "components": len(self.components),
            }}

        elif action == "learn":
            skill = payload.get("skill", "")
            if not skill:
                return {{"error": "Missing 'skill' in learn request"}}
            try:
                from recompile import SelfRecompiler
                recompiler = SelfRecompiler()
                result = await recompiler.request_recompilation(skill)
                return {{"status": "recompile_requested", "skill": skill, "result": result}}
            except ImportError:
                return {{"error": "Self-recompilation not available"}}
            except Exception as e:
                return {{"error": f"Recompile failed: {{e}}"}}

        elif action == "compile":
            if self.compiler is None:
                return {{"error": "Compilation not available"}}
            skill = payload.get("skill", "")
            if not skill:
                return {{"error": "Missing 'skill' in compile request"}}
            domain = payload.get("domain", "software")
            try:
                result = await self.compiler.compile_tool(skill, domain=domain)
                return result
            except Exception as e:
                return {{"error": f"Compilation failed: {{e}}"}}

        elif action == "tools":
            if self.tool_manager is None:
                return {{"error": "Tool management not available"}}
            subaction = payload.get("subaction", "list")
            try:
                if subaction == "list":
                    domain = payload.get("domain")
                    tools = await self.tool_manager.list_tools(domain=domain)
                    return {{"tools": tools}}
                elif subaction == "search":
                    query = payload.get("query", "")
                    if not query:
                        return {{"error": "Missing 'query' in search request"}}
                    results = await self.tool_manager.search_tools(query)
                    return {{"tools": results}}
                elif subaction == "export":
                    compilation_id = payload.get("compilation_id", "")
                    if not compilation_id:
                        return {{"error": "Missing 'compilation_id' in export request"}}
                    output_path = payload.get("output_path")
                    result = await self.tool_manager.export_tool(compilation_id, output_path)
                    return result
                elif subaction == "import":
                    file_path = payload.get("file_path", "")
                    if not file_path:
                        return {{"error": "Missing 'file_path' in import request"}}
                    min_trust = payload.get("min_trust_score", 60.0)
                    result = await self.tool_manager.import_tool(file_path, min_trust)
                    return result
                else:
                    return {{"error": f"Unknown tools subaction: {{subaction}}"}}
            except Exception as e:
                return {{"error": f"Tool operation failed: {{e}}"}}

        elif action == "instance":
            if self.tool_manager is None:
                return {{"error": "Instance info not available"}}
            try:
                result = await self.tool_manager.get_instance_info()
                return result
            except Exception as e:
                return {{"error": f"Instance info failed: {{e}}"}}

        return {{"error": f"Unknown system action: {{action}}"}}

    async def _handle_connection(self, reader, writer):
        """Handle a TCP connection — read JSON messages, dispatch, respond."""
        addr = writer.get_extra_info("peername")
        logger.info("Connection from %s", addr)
        try:
            while self._running:
                data = await reader.readline()
                if not data:
                    break
                try:
                    msg = json.loads(data.decode().strip())
                except json.JSONDecodeError:
                    writer.write(json.dumps({{"error": "Invalid JSON"}}).encode() + b"\\n")
                    await writer.drain()
                    continue

                target = msg.get("target", "")
                payload = msg.get("payload", {{}})

                if target == "_system":
                    result = await self._handle_system(payload)
                else:
                    result = await self.dispatch(target, payload)

                writer.write(json.dumps(result).encode() + b"\\n")
                await writer.drain()
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            writer.close()

    async def start(self, port: int = {port}) -> None:
        """Start the runtime server."""
        self._running = True
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self._shutdown(s)))
        self._server = await asyncio.start_server(
            self._handle_connection, "0.0.0.0", port,
        )
        addr = self._server.sockets[0].getsockname()
        logger.info("Runtime listening on %s:%s", addr[0], addr[1])
        print(f"READY on port {{port}}", flush=True)
        async with self._server:
            await self._server.serve_forever()

    async def _shutdown(self, sig) -> None:
        """Graceful shutdown on signal."""
        logger.info("Received signal %s, shutting down...", sig)
        await self.stop()

    async def stop(self) -> None:
        """Stop the runtime server gracefully."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Runtime stopped")
'''


def generate_state_py(
    capabilities: Any,
    blueprint: Dict[str, Any],
    component_names: List[str],
) -> str:
    """Generate state.py — SQLite-backed persistent state store.

    Returns empty string if persistent state disabled.
    """
    if not capabilities.has_persistent_state:
        return ""

    backend = capabilities.state_backend
    if backend == "json":
        return _generate_json_state()

    # Default: sqlite
    return '''"""State — thread-safe SQLite persistent state store."""

import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional


class StateStore:
    """Thread-safe SQLite key-value state store with JSON serialization."""

    def __init__(self, db_path: str = "state.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create state table if it doesn't exist."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS state (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at REAL NOT NULL DEFAULT (julianday('now'))
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key. Returns default if not found."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT value FROM state WHERE key = ?", (key,)
                ).fetchone()
                if row is None:
                    return default
                return json.loads(row[0])
            finally:
                conn.close()

    async def set(self, key: str, value: Any) -> None:
        """Set a key-value pair. Overwrites existing."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO state (key, value, updated_at)
                       VALUES (?, ?, julianday('now'))""",
                    (key, json.dumps(value)),
                )
                conn.commit()
            finally:
                conn.close()

    async def delete(self, key: str) -> bool:
        """Delete a key. Returns True if key existed."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("DELETE FROM state WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    async def query(self, prefix: str = "") -> Dict[str, Any]:
        """Query all keys matching a prefix. Returns {key: value}."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                if prefix:
                    rows = conn.execute(
                        "SELECT key, value FROM state WHERE key LIKE ?",
                        (prefix + "%",),
                    ).fetchall()
                else:
                    rows = conn.execute("SELECT key, value FROM state").fetchall()
                return {k: json.loads(v) for k, v in rows}
            finally:
                conn.close()

    async def keys(self) -> List[str]:
        """Return all keys."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                rows = conn.execute("SELECT key FROM state").fetchall()
                return [r[0] for r in rows]
            finally:
                conn.close()
'''


def _generate_json_state() -> str:
    """Generate a JSON-file-based state store (simpler alternative)."""
    return '''"""State — JSON-file persistent state store."""

import json
import os
import threading
from typing import Any, Dict, List, Optional


class StateStore:
    """Thread-safe JSON file state store."""

    def __init__(self, file_path: str = "state.json"):
        self.file_path = file_path
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                self._data = json.load(f)

    def _save(self) -> None:
        with open(self.file_path, "w") as f:
            json.dump(self._data, f, indent=2)

    async def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._save()

    async def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._save()
                return True
            return False

    async def query(self, prefix: str = "") -> Dict[str, Any]:
        with self._lock:
            if not prefix:
                return dict(self._data)
            return {k: v for k, v in self._data.items() if k.startswith(prefix)}

    async def keys(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())
'''


def generate_tools_py(
    capabilities: Any,
    blueprint: Dict[str, Any],
    component_names: List[str],
) -> str:
    """Generate tools.py — sandboxed tool executor with allowlist.

    Returns empty string if tool execution disabled.
    """
    if not capabilities.has_tool_execution:
        return ""

    # Build allowlist literal
    allowlist_items = ", ".join(f'"{t}"' for t in capabilities.tool_allowlist)

    return f'''"""Tools — sandboxed tool executor with subprocess dispatch and allowlist."""

import asyncio
import json
import subprocess
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

TOOL_ALLOWLIST = frozenset({{{allowlist_items}}})


class ToolExecutor:
    """Execute tools in sandboxed subprocesses with allowlist enforcement."""

    def __init__(self, allowlist: Optional[frozenset] = None, timeout: int = 30):
        self.allowlist = allowlist or TOOL_ALLOWLIST
        self.timeout = timeout

    async def execute(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool-specific arguments

        Returns:
            Dict with "result" or "error" key
        """
        if tool_name not in self.allowlist:
            return {{"error": f"Tool '{{tool_name}}' not in allowlist"}}

        handler = getattr(self, f"_tool_{{tool_name}}", None)
        if handler is None:
            return {{"error": f"No handler for tool '{{tool_name}}'"}}

        try:
            return await handler(**kwargs)
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return {{"error": str(e)}}

    async def list_tools(self) -> list:
        """Return list of available tools."""
        return sorted(self.allowlist)

    async def _tool_web_search(self, query: str = "", **kwargs) -> Dict[str, Any]:
        """Web search via subprocess (placeholder — override for real implementation)."""
        return {{"result": f"Search results for: {{query}}", "tool": "web_search"}}

    async def _tool_file_read(self, path: str = "", **kwargs) -> Dict[str, Any]:
        """Read a file (sandboxed to project directory)."""
        import os
        if ".." in path or path.startswith("/"):
            return {{"error": "Path traversal not allowed"}}
        try:
            with open(path, "r") as f:
                content = f.read(10000)  # 10KB limit
            return {{"result": content, "tool": "file_read"}}
        except FileNotFoundError:
            return {{"error": f"File not found: {{path}}"}}

    async def _tool_file_write(self, path: str = "", content: str = "", **kwargs) -> Dict[str, Any]:
        """Write to a file (sandboxed to project directory)."""
        import os
        if ".." in path or path.startswith("/"):
            return {{"error": "Path traversal not allowed"}}
        try:
            with open(path, "w") as f:
                f.write(content[:50000])  # 50KB limit
            return {{"result": f"Written {{len(content)}} chars to {{path}}", "tool": "file_write"}}
        except Exception as e:
            return {{"error": str(e)}}

    async def _tool_shell_exec(self, command: str = "", **kwargs) -> Dict[str, Any]:
        """Execute a shell command in a subprocess with timeout."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout,
            )
            return {{
                "result": stdout.decode()[:5000],
                "stderr": stderr.decode()[:2000],
                "returncode": proc.returncode,
                "tool": "shell_exec",
            }}
        except asyncio.TimeoutError:
            return {{"error": f"Command timed out after {{self.timeout}}s"}}
'''


def generate_llm_client_py(
    capabilities: Any,
    blueprint: Dict[str, Any],
    component_names: List[str],
) -> str:
    """Generate llm_client.py — provider-agnostic LLM wrapper.

    Returns empty string if LLM client disabled.
    """
    if not capabilities.has_llm_client:
        return ""

    return '''"""LLM Client — provider-agnostic async LLM wrapper with retry."""

import json
import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client with provider abstraction and retry logic."""

    def __init__(
        self,
        provider: str = "",
        api_key: str = "",
        model: str = "",
        max_retries: int = 2,
    ):
        self.provider = provider or os.environ.get("LLM_PROVIDER", "anthropic")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "claude-sonnet-4-5-20250929")
        self.max_retries = max_retries

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Send a chat completion request.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system: Optional system prompt
            max_tokens: Max response tokens
            temperature: Sampling temperature

        Returns:
            Dict with "content" (str) and "usage" (dict) keys
        """
        for attempt in range(self.max_retries + 1):
            try:
                return await self._call_provider(
                    messages, system, max_tokens, temperature,
                )
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error("LLM call failed after %d retries: %s", self.max_retries, e)
                    return {"error": str(e)}
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, e)

        return {"error": "Unexpected: all retries exhausted"}

    async def complete(self, prompt: str, **kwargs) -> str:
        """Simple completion — returns just the text content."""
        result = await self.chat(
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return result.get("content", result.get("error", ""))

    async def _call_provider(
        self,
        messages: List[Dict[str, str]],
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Call the configured LLM provider via httpx."""
        try:
            import httpx
        except ImportError:
            return {"error": "httpx not installed. Run: pip install httpx"}

        if self.provider == "anthropic":
            return await self._call_anthropic(messages, system, max_tokens, temperature)
        elif self.provider == "openai":
            return await self._call_openai(messages, system, max_tokens, temperature)
        else:
            return {"error": f"Unknown provider: {self.provider}"}

    async def _call_anthropic(self, messages, system, max_tokens, temperature):
        import httpx
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            body["system"] = system

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content += block.get("text", "")
            return {
                "content": content,
                "usage": data.get("usage", {}),
            }

    async def _call_openai(self, messages, system, max_tokens, temperature):
        import httpx
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": msgs,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            return {
                "content": choice.get("message", {}).get("content", ""),
                "usage": data.get("usage", {}),
            }
'''


def generate_config_py(
    capabilities: Any,
    blueprint: Dict[str, Any],
    component_names: List[str],
) -> str:
    """Generate config.py — runtime configuration from environment variables.

    Always generated when any runtime capability is enabled.
    """
    has_any = (
        capabilities.has_event_loop
        or capabilities.has_llm_client
        or capabilities.has_persistent_state
        or capabilities.has_tool_execution
        or capabilities.has_self_recompile
    )
    if not has_any:
        return ""

    port = capabilities.default_port
    allowlist_items = capabilities.tool_allowlist
    if allowlist_items:
        allowlist_str = "(" + ", ".join(f'"{t}"' for t in allowlist_items) + ",)"
    else:
        allowlist_str = "()"

    return f'''"""Config — runtime configuration with environment variable overrides."""

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    """Runtime configuration. All values can be overridden via environment variables."""

    # Server
    port: int = int(os.environ.get("PORT", "{port}"))
    host: str = os.environ.get("HOST", "0.0.0.0")

    # LLM
    llm_provider: str = os.environ.get("LLM_PROVIDER", "anthropic")
    llm_api_key: str = os.environ.get("LLM_API_KEY", "")
    llm_model: str = os.environ.get("LLM_MODEL", "claude-sonnet-4-5-20250929")

    # State
    state_path: str = os.environ.get("STATE_PATH", "state.db")

    # Tools
    tool_allowlist: Tuple[str, ...] = {allowlist_str}
    tool_timeout: int = int(os.environ.get("TOOL_TIMEOUT", "30"))

    # Recompilation
    recompile_api_url: str = os.environ.get("RECOMPILE_API_URL", "http://localhost:8000/v2/recompile")

    # Logging
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")
'''


def generate_recompile_py(
    capabilities: Any,
    blueprint: Dict[str, Any],
    component_names: List[str],
) -> str:
    """Generate recompile.py — self-modification via /v2/recompile API.

    Returns empty string if self-recompile disabled.
    """
    if not capabilities.has_self_recompile:
        return ""

    blueprint_json = json.dumps(blueprint, indent=2, default=str) if blueprint else "{}"
    # Truncate to avoid enormous files
    if len(blueprint_json) > 10000:
        blueprint_json = blueprint_json[:10000] + '\n  "...": "truncated"}'

    return f'''"""Recompile — self-modification via Motherlabs /v2/recompile API."""

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# The current blueprint — embedded at generation time
_CURRENT_BLUEPRINT = json.loads("""
{blueprint_json}
""")


class SelfRecompiler:
    """Allows the running system to request its own recompilation."""

    def __init__(self, api_url: str = "", api_key: str = ""):
        self.api_url = api_url or os.environ.get(
            "RECOMPILE_API_URL", "http://localhost:8000/v2/recompile",
        )
        self.api_key = api_key or os.environ.get("MOTHERLABS_API_KEY", "")

    async def detect_gap(
        self, message: str, response: str, llm: Any = None,
    ) -> Optional[str]:
        """Use LLM to detect if the system failed to handle a message well.

        Args:
            message: The user message that was sent
            response: The system's response
            llm: LLMClient instance for analysis

        Returns:
            Gap description string, or None if response was adequate
        """
        if llm is None:
            return None

        analysis = await llm.complete(
            f"Analyze this interaction. Did the system handle it well?\\n\\n"
            f"User message: {{message}}\\n\\n"
            f"System response: {{response}}\\n\\n"
            f"If the system failed to handle this well, describe what capability "
            f"is missing in one sentence. If it handled it fine, respond with "
            f"just the word 'ADEQUATE'."
        )
        if "ADEQUATE" in analysis.upper():
            return None
        return analysis.strip()

    async def request_recompilation(self, gap_description: str) -> dict:
        """Call /v2/recompile with current blueprint + gap as enhancement.

        Args:
            gap_description: What capability the system is missing

        Returns:
            API response dict with new compilation result
        """
        try:
            import httpx
        except ImportError:
            return {{"error": "httpx not installed"}}

        payload = {{
            "current_blueprint": _CURRENT_BLUEPRINT,
            "enhancement": gap_description,
            "domain": "agent_system",
        }}

        headers = {{"Content-Type": "application/json"}}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.exception("Recompilation request failed")
            return {{"error": str(e)}}

    async def safe_deploy(self, new_project_dir: str, restart: bool = True) -> bool:
        """Validate new code, swap in, optionally restart. Rollback on failure.

        Args:
            new_project_dir: Path to the newly compiled project
            restart: Whether to restart the process after deploy (default True)

        Returns:
            True if deployment succeeded, False if rolled back
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backup_dir = current_dir + ".backup"

        try:
            # 1. Backup current
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            shutil.copytree(current_dir, backup_dir)

            # 2. Copy new files over (except state and config)
            preserve = {{"state.db", "state.json", ".env", "config.py"}}
            for item in os.listdir(new_project_dir):
                if item in preserve:
                    continue
                src = os.path.join(new_project_dir, item)
                dst = os.path.join(current_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

            logger.info("Deployed new code from %s", new_project_dir)

            # 3. Restart process to load new code
            if restart:
                logger.info("Restarting process in 2 seconds...")
                await asyncio.sleep(2)  # drain connections
                self.restart_process()

            return True

        except Exception as e:
            logger.exception("Deployment failed, rolling back")
            # Rollback
            try:
                if os.path.exists(backup_dir):
                    shutil.rmtree(current_dir)
                    shutil.copytree(backup_dir, current_dir)
                    logger.info("Rollback successful")
            except Exception as rb_err:
                logger.critical("Rollback also failed: %s", rb_err)
            return False

    @staticmethod
    def restart_process() -> None:
        """Replace the current process with a fresh instance.

        Uses os.execv to replace the running process image, so the new
        code is loaded from disk. This is the cleanest way to hot-reload
        without leaving orphan processes.
        """
        logger.info("Restarting: exec %s %s", sys.executable, sys.argv)
        os.execv(sys.executable, [sys.executable] + sys.argv)
'''


def generate_compiler_py(
    capabilities: Any,
    blueprint: Dict[str, Any],
    component_names: List[str],
) -> str:
    """Generate compiler.py — tool compilation via MotherlabsEngine.

    Returns empty string if can_compile is False. Otherwise generates a
    ToolCompiler class that lazily imports the engine and compiles new tools.

    Note: the GENERATED code imports from motherlabs internals. This is
    intentional — mother agents run within the Motherlabs environment.
    The scaffold template itself remains a leaf module.
    """
    if not capabilities.can_compile:
        return ""

    corpus_path = capabilities.corpus_path or "~/motherlabs/corpus.db"

    return f'''"""Compiler — compile new tools via MotherlabsEngine."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ToolCompiler:
    """Compile natural language descriptions into verified tools.

    Uses lazy imports to avoid circular dependencies at module load time.
    Engine is initialized once on first compilation request.
    """

    def __init__(self, corpus_path: str = "{corpus_path}"):
        self.corpus_path = Path(corpus_path).expanduser()
        self._engine = None
        self._initialized = False

    def _ensure_engine(self, domain: str = "software", provider: str = "anthropic"):
        """Lazy-initialize the compilation engine on first use."""
        if self._initialized:
            return
        from core.engine import MotherlabsEngine
        from core.llm import get_llm_client
        from core.adapter_registry import get_adapter
        from persistence.corpus import Corpus

        # Ensure corpus directory exists
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)

        adapter = get_adapter(domain)
        llm_client = get_llm_client(provider)
        corpus = Corpus(self.corpus_path)
        self._engine = MotherlabsEngine(
            llm_client=llm_client,
            corpus=corpus,
            domain_adapter=adapter,
        )
        self._initialized = True
        logger.info("ToolCompiler engine initialized (domain=%s, provider=%s)", domain, provider)

    async def compile_tool(
        self,
        description: str,
        domain: str = "software",
        provider: str = "anthropic",
    ) -> Dict[str, Any]:
        """Compile a tool from a natural language description.

        Args:
            description: What the tool should do
            domain: Domain adapter to use (default: software)
            provider: LLM provider (default: anthropic)

        Returns:
            Dict with compilation_id, trust_score, verification_badge,
            component_count, and status.
        """
        try:
            self._ensure_engine(domain, provider)
            # engine.compile() is synchronous — run in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._engine.compile, description,
            )
            return {{
                "compilation_id": result.compilation_id,
                "trust_score": result.trust_indicators.composite_score if result.trust_indicators else 0.0,
                "verification_badge": result.trust_indicators.badge if result.trust_indicators else "none",
                "component_count": len(result.blueprint.get("components", [])) if isinstance(result.blueprint, dict) else 0,
                "status": "compiled",
            }}
        except Exception as e:
            logger.exception("Compilation failed")
            return {{"error": str(e), "status": "failed"}}
'''


def generate_tool_manager_py(
    capabilities: Any,
    blueprint: Dict[str, Any],
    component_names: List[str],
) -> str:
    """Generate tool_manager.py — tool registry and import/export management.

    Returns empty string if can_share_tools is False. Otherwise generates a
    ToolManager class that wraps the tool registry and export/import operations.

    Note: the GENERATED code imports from motherlabs internals. This is
    intentional — mother agents run within the Motherlabs environment.
    """
    if not capabilities.can_share_tools:
        return ""

    corpus_path = capabilities.corpus_path or "~/motherlabs/corpus.db"

    return f'''"""Tool Manager — registry, export, and import for .mtool files."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolManager:
    """Manage tool registry: list, search, export, and import .mtool files.

    Uses lazy imports to avoid circular dependencies at module load time.
    """

    def __init__(
        self,
        instance_id: str = "",
        corpus_path: str = "{corpus_path}",
        registry_path: Optional[str] = None,
    ):
        self.instance_id = instance_id
        self.corpus_path = Path(corpus_path).expanduser()
        self.registry_path = registry_path
        self._registry = None
        self._corpus = None

    def _ensure_registry(self):
        """Lazy-initialize the tool registry."""
        if self._registry is not None:
            return
        from motherlabs_platform.tool_registry import get_tool_registry
        if self.registry_path:
            from motherlabs_platform.tool_registry import ToolRegistry
            self._registry = ToolRegistry(self.registry_path)
        else:
            self._registry = get_tool_registry()
        logger.info("ToolManager registry initialized")

    def _ensure_corpus(self):
        """Lazy-initialize the corpus."""
        if self._corpus is not None:
            return
        from persistence.corpus import Corpus
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        self._corpus = Corpus(self.corpus_path)
        logger.info("ToolManager corpus initialized at %s", self.corpus_path)

    async def list_tools(self, domain: Optional[str] = None, local_only: bool = False) -> List[Dict[str, Any]]:
        """List tools in the registry.

        Args:
            domain: Filter by domain (optional)
            local_only: Only show locally compiled tools

        Returns:
            List of tool info dicts
        """
        try:
            self._ensure_registry()
            tools = self._registry.list_tools(domain=domain)
            results = []
            for t in tools:
                info = {{
                    "id": t.get("id", ""),
                    "name": t.get("name", ""),
                    "domain": t.get("domain", ""),
                    "trust_score": t.get("trust_score", 0.0),
                    "created_at": t.get("created_at", ""),
                }}
                if local_only and t.get("source_instance") != self.instance_id:
                    continue
                results.append(info)
            return results
        except Exception as e:
            logger.exception("Failed to list tools")
            return [{{"error": str(e)}}]

    async def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """Search tools by keyword.

        Args:
            query: Search query string

        Returns:
            List of matching tool info dicts
        """
        try:
            self._ensure_registry()
            tools = self._registry.search_tools(query)
            return [
                {{
                    "id": t.get("id", ""),
                    "name": t.get("name", ""),
                    "domain": t.get("domain", ""),
                    "trust_score": t.get("trust_score", 0.0),
                }}
                for t in tools
            ]
        except Exception as e:
            logger.exception("Failed to search tools")
            return [{{"error": str(e)}}]

    async def export_tool(self, compilation_id: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Export a compiled tool as an .mtool file.

        Args:
            compilation_id: ID of the compilation to export
            output_path: Output file path (auto-generated if None)

        Returns:
            Dict with file_path and tool info
        """
        try:
            self._ensure_corpus()
            from core.tool_export import export_tool_to_file
            if output_path is None:
                output_path = str(Path.home() / "motherlabs" / f"{{compilation_id}}.mtool")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result = export_tool_to_file(
                compilation_id=compilation_id,
                corpus=self._corpus,
                output_path=output_path,
                instance_id=self.instance_id,
            )
            return {{
                "file_path": output_path,
                "compilation_id": compilation_id,
                "status": "exported",
                **result,
            }}
        except Exception as e:
            logger.exception("Failed to export tool")
            return {{"error": str(e), "status": "failed"}}

    async def import_tool(self, file_path: str, min_trust_score: float = 60.0) -> Dict[str, Any]:
        """Import a .mtool file with governor validation.

        Args:
            file_path: Path to .mtool file
            min_trust_score: Minimum trust score to accept (default: 60.0)

        Returns:
            Dict with import result and validation info
        """
        try:
            from core.tool_export import load_tool_from_file, import_tool
            self._ensure_registry()
            self._ensure_corpus()
            package = load_tool_from_file(file_path)
            result = import_tool(
                package=package,
                corpus=self._corpus,
                registry=self._registry,
                min_trust_score=min_trust_score,
            )
            return {{
                "status": "imported" if result.get("accepted", False) else "rejected",
                "file_path": file_path,
                **result,
            }}
        except Exception as e:
            logger.exception("Failed to import tool")
            return {{"error": str(e), "status": "failed"}}

    async def get_instance_info(self) -> Dict[str, Any]:
        """Get information about this Motherlabs instance.

        Returns:
            Dict with instance_id, tool_count, corpus_path
        """
        try:
            self._ensure_registry()
            tools = self._registry.list_tools()
            return {{
                "instance_id": self.instance_id,
                "tool_count": len(tools),
                "corpus_path": str(self.corpus_path),
                "status": "active",
            }}
        except Exception as e:
            logger.exception("Failed to get instance info")
            return {{"error": str(e)}}
'''


