"""
Peer WebSocket — real-time push events between Mother instances.

Provides PeerWSServer (broadcasts events) and PeerWSClient (listens).
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeerEvent:
    """A real-time event exchanged between peers."""
    event_type: str  # "tool_published" | "compilation_complete" | "peer_joined" | "peer_left"
    instance_id: str
    payload: Dict[str, Any]
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "PeerEvent":
        d = json.loads(data)
        return cls(**d)


class PeerWSServer:
    """WebSocket server that broadcasts peer events to connected clients."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
        self._host = host
        self._port = port
        self._clients: Set = set()
        self._server = None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def start(self) -> None:
        """Start the WebSocket server."""
        import websockets
        self._server = await websockets.serve(
            self._handler,
            self._host,
            self._port,
        )
        logger.info(f"PeerWSServer listening on ws://{self._host}:{self._port}")

    async def stop(self) -> None:
        """Stop the WebSocket server and disconnect all clients."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Close all connected clients
        if self._clients:
            await asyncio.gather(
                *[ws.close() for ws in self._clients],
                return_exceptions=True,
            )
            self._clients.clear()

        logger.info("PeerWSServer stopped")

    async def broadcast(self, event: PeerEvent) -> int:
        """Broadcast an event to all connected clients.

        Returns number of clients that received the message.
        """
        if not self._clients:
            return 0

        message = event.to_json()
        disconnected = set()
        sent = 0

        for ws in self._clients:
            try:
                await ws.send(message)
                sent += 1
            except Exception:
                disconnected.add(ws)

        # Cleanup disconnected clients
        self._clients -= disconnected
        return sent

    async def _handler(self, websocket, path=None) -> None:
        """Handle a new WebSocket connection."""
        self._clients.add(websocket)
        logger.debug(f"Peer WS client connected ({self.client_count} total)")
        try:
            async for message in websocket:
                # Clients don't send messages in this protocol, but drain anyway
                pass
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            logger.debug(f"Peer WS client disconnected ({self.client_count} total)")


class PeerWSClient:
    """WebSocket client that connects to a remote PeerWSServer."""

    def __init__(
        self,
        ws_url: str,
        on_event: Optional[Callable[[PeerEvent], None]] = None,
    ):
        self._ws_url = ws_url
        self._on_event = on_event
        self._ws = None
        self._listen_task: Optional[asyncio.Task] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Connect to the remote WebSocket server."""
        import websockets
        try:
            self._ws = await websockets.connect(self._ws_url)
            self._connected = True
            self._listen_task = asyncio.create_task(self._listen_loop())
            logger.info(f"Connected to peer WS at {self._ws_url}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to {self._ws_url}: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from the remote server."""
        self._connected = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _listen_loop(self) -> None:
        """Listen for events from the server."""
        try:
            async for message in self._ws:
                try:
                    event = PeerEvent.from_json(message)
                    if self._on_event:
                        self._on_event(event)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"Invalid peer event: {e}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"WS listen loop ended: {e}")
        finally:
            self._connected = False
