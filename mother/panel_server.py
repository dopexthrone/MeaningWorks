"""
Panel server — HTTP/WebSocket adapter for the Mother panel.

Wraps EngineBridge as a local IPC server on 127.0.0.1:7770.
SwiftUI panel (or any client) connects via HTTP + WebSocket.

Does NOT import from core/. Uses bridge.py as the only engine seam.

CLI: python -m mother.panel_server --port 7770
"""

import asyncio
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Set

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from mother.bridge import EngineBridge
from mother.config import load_config, save_config, MotherConfig, DEFAULT_CONFIG_DIR
from mother.panel_protocol import PanelMessage, MessageType, Channel
from mother.design_tokens import export_json as export_design_tokens

logger = logging.getLogger("mother.panel_server")

# --- Paths ---

TOKEN_PATH = DEFAULT_CONFIG_DIR / "panel.token"
PID_PATH = DEFAULT_CONFIG_DIR / "panel-server.pid"
DESIGN_TOKENS_PATH = DEFAULT_CONFIG_DIR / "design-tokens.json"
DB_PATH = DEFAULT_CONFIG_DIR / "mother.db"


# --- PID lockfile ---

def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_lock(pid_path: Optional[Path] = None) -> bool:
    pid_path = pid_path or PID_PATH
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    if pid_path.exists():
        try:
            stored_pid = int(pid_path.read_text().strip())
            if _is_pid_alive(stored_pid):
                return False
        except (ValueError, OSError):
            pass

    pid_path.write_text(str(os.getpid()))
    return True


def release_lock(pid_path: Optional[Path] = None) -> None:
    pid_path = pid_path or PID_PATH
    try:
        if pid_path.exists():
            stored_pid = int(pid_path.read_text().strip())
            if stored_pid == os.getpid():
                pid_path.unlink()
    except (ValueError, OSError):
        pass


# --- Auth token ---

def generate_auth_token() -> str:
    """Generate random auth token and write to ~/.motherlabs/panel.token."""
    token = secrets.token_hex(32)
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(token)
    # Restrict permissions (owner-only)
    TOKEN_PATH.chmod(0o600)
    return token


def load_auth_token() -> Optional[str]:
    """Load existing token from disk."""
    if TOKEN_PATH.exists():
        return TOKEN_PATH.read_text().strip()
    return None


# --- Server state ---

class ServerState:
    """Mutable state shared across routes and WebSocket handlers."""

    def __init__(self, bridge: EngineBridge, auth_token: str):
        self.bridge = bridge
        self.auth_token = auth_token
        self.start_time = time.monotonic()
        self.ws_clients: Set[WebSocket] = set()
        self.subscriptions: dict = {}  # ws -> set of channels
        self.senses_task: Optional[asyncio.Task] = None


# --- HTTP route handlers ---

async def health(request: Request) -> JSONResponse:
    state: ServerState = request.app.state.server
    uptime = time.monotonic() - state.start_time
    return JSONResponse({"alive": True, "uptime_s": round(uptime, 2)})


async def status(request: Request) -> JSONResponse:
    state: ServerState = request.app.state.server
    if not _check_auth(request, state):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return JSONResponse(state.bridge.get_status())


async def get_config(request: Request) -> JSONResponse:
    state: ServerState = request.app.state.server
    if not _check_auth(request, state):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    cfg = load_config()
    return JSONResponse(asdict(cfg))


async def post_config(request: Request) -> JSONResponse:
    state: ServerState = request.app.state.server
    if not _check_auth(request, state):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        data = await request.json()
        cfg = load_config()
        known_fields = {f for f in cfg.__dataclass_fields__}
        for k, v in data.items():
            if k in known_fields:
                object.__setattr__(cfg, k, v)
        save_config(cfg)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def post_chat(request: Request) -> JSONResponse:
    state: ServerState = request.app.state.server
    if not _check_auth(request, state):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        data = await request.json()
        messages = data.get("messages", [])
        system_prompt = data.get("system_prompt", "")
        result = await state.bridge.chat(messages, system_prompt)
        return JSONResponse({"text": result, "cost": state.bridge.get_last_call_cost()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def post_compile(request: Request) -> JSONResponse:
    state: ServerState = request.app.state.server
    if not _check_auth(request, state):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        data = await request.json()
        description = data.get("description", "")
        result = await state.bridge.compile(description)
        # CompileResult is a dataclass — convert to dict
        return JSONResponse(asdict(result))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_tools(request: Request) -> JSONResponse:
    state: ServerState = request.app.state.server
    if not _check_auth(request, state):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        tools = await state.bridge.list_tools()
        return JSONResponse(tools)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_appendages(request: Request) -> JSONResponse:
    state: ServerState = request.app.state.server
    if not _check_auth(request, state):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        appendages = await state.bridge.get_active_appendages(str(DB_PATH))
        return JSONResponse(appendages)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def post_file_search(request: Request) -> JSONResponse:
    state: ServerState = request.app.state.server
    if not _check_auth(request, state):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        data = await request.json()
        results = await state.bridge.search_files(
            data.get("query", ""), data.get("path"),
        )
        return JSONResponse(results)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def post_file_read(request: Request) -> JSONResponse:
    state: ServerState = request.app.state.server
    if not _check_auth(request, state):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        data = await request.json()
        result = await state.bridge.read_file(data.get("path", ""))
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_goals(request: Request) -> JSONResponse:
    state: ServerState = request.app.state.server
    if not _check_auth(request, state):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        goals = await state.bridge.get_active_goals(str(DB_PATH))
        return JSONResponse(goals)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def post_goal(request: Request) -> JSONResponse:
    state: ServerState = request.app.state.server
    if not _check_auth(request, state):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        data = await request.json()
        result = await state.bridge.add_goal(
            str(DB_PATH), data.get("description", ""),
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# --- Auth check ---

def _check_auth(request: Request, state: ServerState) -> bool:
    """Verify Bearer token from Authorization header."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:] == state.auth_token
    return False


# --- WebSocket handler ---

async def ws_endpoint(websocket: WebSocket):
    state: ServerState = websocket.app.state.server
    await websocket.accept()

    # First message must be auth
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        msg = PanelMessage.from_json(raw)
        if msg.msg_type != MessageType.AUTH or msg.payload.get("token") != state.auth_token:
            await _ws_send(websocket, PanelMessage(
                msg_type=MessageType.ERROR, payload={"message": "unauthorized"},
            ))
            await websocket.close(code=4001)
            return
    except (asyncio.TimeoutError, ValueError, WebSocketDisconnect):
        await websocket.close(code=4001)
        return

    # Authenticated — register client
    state.ws_clients.add(websocket)
    state.subscriptions[websocket] = set()

    try:
        # Send ready
        await _ws_send(websocket, PanelMessage(
            msg_type=MessageType.READY, payload={"version": "1.0"},
        ))

        # Message loop
        while True:
            raw = await websocket.receive_text()
            msg = PanelMessage.from_json(raw)
            await _handle_ws_message(websocket, msg, state)

    except (WebSocketDisconnect, Exception):
        pass
    finally:
        state.ws_clients.discard(websocket)
        state.subscriptions.pop(websocket, None)


async def _ws_send(ws: WebSocket, msg: PanelMessage) -> None:
    """Send a PanelMessage over WebSocket. Silently drops on error."""
    try:
        await ws.send_text(msg.to_json())
    except Exception:
        pass


async def _handle_ws_message(ws: WebSocket, msg: PanelMessage, state: ServerState) -> None:
    """Dispatch incoming WebSocket message."""
    if msg.msg_type == MessageType.SUBSCRIBE:
        channels = msg.payload.get("channels", [])
        valid = {c for c in channels if c in Channel.ALL}
        state.subscriptions[ws] = valid
        return

    if msg.msg_type == MessageType.CHAT:
        asyncio.create_task(_handle_ws_chat(ws, msg, state))
        return

    if msg.msg_type == MessageType.CANCEL_STREAM:
        state.bridge.cancel_chat_stream()
        return

    if msg.msg_type == MessageType.CAPTURE_SCREEN:
        asyncio.create_task(_handle_ws_capture(ws, msg, state))
        return

    # Unknown message type
    await _ws_send(ws, PanelMessage(
        msg_type=MessageType.ERROR,
        msg_id=msg.msg_id,
        payload={"message": f"Unknown message type: {msg.msg_type}"},
    ))


async def _handle_ws_chat(ws: WebSocket, msg: PanelMessage, state: ServerState) -> None:
    """Stream chat response back via WebSocket."""
    messages = msg.payload.get("messages", [])
    system_prompt = msg.payload.get("system_prompt", "")

    try:
        state.bridge.begin_chat_stream()
        stream_task = asyncio.create_task(
            state.bridge.stream_chat(messages, system_prompt)
        )

        async for token in state.bridge.stream_chat_tokens():
            await _ws_send(ws, PanelMessage(
                msg_type=MessageType.CHAT_TOKEN,
                msg_id=msg.msg_id,
                payload={"token": token},
            ))

        await stream_task

        full_text = state.bridge.get_stream_result() or ""
        cost = state.bridge.get_last_call_cost()
        await _ws_send(ws, PanelMessage(
            msg_type=MessageType.CHAT_DONE,
            msg_id=msg.msg_id,
            payload={"full_text": full_text, "cost": cost},
        ))

    except Exception as e:
        await _ws_send(ws, PanelMessage(
            msg_type=MessageType.ERROR,
            msg_id=msg.msg_id,
            payload={"message": str(e)},
        ))


async def _handle_ws_capture(ws: WebSocket, msg: PanelMessage, state: ServerState) -> None:
    """Handle screen capture request."""
    try:
        from mother.screen import ScreenCaptureBridge
        bridge = ScreenCaptureBridge(enabled=True)
        result = await asyncio.to_thread(bridge.capture)
        await _ws_send(ws, PanelMessage(
            msg_type=MessageType.CHAT_DONE,
            msg_id=msg.msg_id,
            payload=result if isinstance(result, dict) else {"result": str(result)},
        ))
    except Exception as e:
        await _ws_send(ws, PanelMessage(
            msg_type=MessageType.ERROR,
            msg_id=msg.msg_id,
            payload={"message": str(e)},
        ))


# --- Senses push loop ---

async def _senses_push_loop(state: ServerState, interval: float = 2.0) -> None:
    """Background task: push senses/posture to subscribed clients."""
    from mother.senses import (
        SenseObservations, compute_senses, compute_posture,
    )

    while True:
        await asyncio.sleep(interval)

        # Gather current observations from bridge state
        bridge = state.bridge
        obs = SenseObservations(
            session_cost=bridge.get_session_cost(),
            cost_limit=load_config().cost_limit,
        )
        senses = compute_senses(obs)
        posture = compute_posture(senses)

        # Push to subscribed clients
        senses_msg = PanelMessage(
            msg_type=MessageType.SENSES_UPDATE,
            payload=asdict(senses),
        )
        posture_msg = PanelMessage(
            msg_type=MessageType.POSTURE_UPDATE,
            payload=asdict(posture),
        )

        for ws, channels in list(state.subscriptions.items()):
            if Channel.SENSES in channels:
                await _ws_send(ws, senses_msg)
                await _ws_send(ws, posture_msg)


# --- App factory ---

def create_app(bridge: Optional[EngineBridge] = None, auth_token: Optional[str] = None) -> Starlette:
    """Create the Starlette ASGI app.

    Accepts optional bridge and auth_token for testing.
    """
    if bridge is None:
        cfg = load_config()
        api_key = cfg.api_keys.get(cfg.provider) or os.environ.get(cfg.get_env_var())
        bridge = EngineBridge(
            provider=cfg.provider,
            model=cfg.get_model(),
            api_key=api_key,
            file_access=cfg.file_access,
            screen_capture_enabled=cfg.screen_capture_enabled,
            microphone_enabled=cfg.microphone_enabled,
        )

    if auth_token is None:
        auth_token = generate_auth_token()

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/status", status, methods=["GET"]),
        Route("/config", get_config, methods=["GET"]),
        Route("/config", post_config, methods=["POST"]),
        Route("/chat", post_chat, methods=["POST"]),
        Route("/compile", post_compile, methods=["POST"]),
        Route("/tools", get_tools, methods=["GET"]),
        Route("/appendages", get_appendages, methods=["GET"]),
        Route("/file/search", post_file_search, methods=["POST"]),
        Route("/file/read", post_file_read, methods=["POST"]),
        Route("/goals", get_goals, methods=["GET"]),
        Route("/goal", post_goal, methods=["POST"]),
        WebSocketRoute("/ws", ws_endpoint),
    ]

    @asynccontextmanager
    async def lifespan(app):
        # Startup
        export_design_tokens(str(DESIGN_TOKENS_PATH))
        logger.info(f"Design tokens written to {DESIGN_TOKENS_PATH}")

        cfg = load_config()
        interval = cfg.panel_senses_push_interval
        app.state.server.senses_task = asyncio.create_task(
            _senses_push_loop(app.state.server, interval)
        )
        logger.info(f"Senses push loop started (interval={interval}s)")

        yield

        # Shutdown
        if app.state.server.senses_task:
            app.state.server.senses_task.cancel()
            try:
                await app.state.server.senses_task
            except asyncio.CancelledError:
                pass

        for ws in list(app.state.server.ws_clients):
            try:
                await ws.close()
            except Exception:
                pass

    app = Starlette(
        routes=routes,
        lifespan=lifespan,
    )
    app.state.server = ServerState(bridge, auth_token)

    return app


# --- CLI entry point ---

def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="mother-panel-server",
        description="Mother panel IPC server.",
    )
    parser.add_argument("--port", type=int, default=7770, help="Port (default: 7770)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    if not acquire_lock():
        print("Panel server is already running.", flush=True)
        raise SystemExit(1)

    try:
        app = create_app()
        token = load_auth_token()
        print(f"Panel server starting on {args.host}:{args.port}", flush=True)
        print(f"Auth token written to {TOKEN_PATH}", flush=True)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
