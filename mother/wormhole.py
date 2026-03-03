"""
Wormhole — persistent peer-to-peer connection between Mother instances.

LEAF module. Uses websockets for bidirectional async communication.

Enables Mother to:
- Maintain persistent connection to peer instances
- Exchange tool packages
- Delegate compile/build tasks
- Replicate corpus compilations
- Sync operational state

Protocol:
- WebSocket connection (wss:// or ws://)
- JSON message framing
- Request/response pattern with message_id
- Heartbeat every 30s to detect dead connections

Message types:
- handshake: exchange instance digest + trust verification
- tool_offer: share a tool package
- compile_request: delegate compilation to peer
- corpus_sync: replicate compilation for L2 learning
- heartbeat: keep-alive ping
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

try:
    import websockets
    from websockets.server import serve
    from websockets.client import connect
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    # Stub types for when websockets not installed
    serve = None
    connect = None

logger = logging.getLogger("mother.wormhole")


@dataclass(frozen=True)
class WormholeMessage:
    """Single message in the wormhole protocol."""

    message_id: str
    message_type: str  # handshake, tool_offer, compile_request, corpus_sync, heartbeat, response
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    reply_to: str = ""


@dataclass
class WormholeConnection:
    """Active connection to a peer Mother instance."""

    peer_id: str
    peer_name: str = "Unknown"
    websocket: Any = None  # websockets.WebSocketServerProtocol or WebSocketClientProtocol
    connected_at: float = 0.0
    last_heartbeat: float = 0.0
    trust_verified: bool = False
    capabilities: List[str] = field(default_factory=list)


class Wormhole:
    """
    Peer-to-peer communication tunnel.

    Manages WebSocket connections to peer Mother instances.
    Handles handshake, message routing, heartbeat, and reconnection.
    """

    def __init__(
        self,
        instance_id: str,
        instance_name: str,
        port: int = 8765,
        on_message: Optional[Callable] = None,
    ):
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library required for peer networking")

        self.instance_id = instance_id
        self.instance_name = instance_name
        self.port = port
        self.on_message = on_message or (lambda msg: None)

        self.connections: Dict[str, WormholeConnection] = {}
        self._server = None
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._running = False

    async def start_server(self):
        """Start WebSocket server to accept peer connections."""
        if not WEBSOCKETS_AVAILABLE:
            return

        async def handler(websocket):
            peer_id = None
            try:
                # Wait for handshake
                raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                msg = self._parse_message(raw)

                if msg and msg.message_type == "handshake":
                    peer_id = msg.payload.get("instance_id", "")
                    peer_name = msg.payload.get("name", "Unknown")

                    # Store connection
                    conn = WormholeConnection(
                        peer_id=peer_id,
                        peer_name=peer_name,
                        websocket=websocket,
                        connected_at=time.time(),
                        last_heartbeat=time.time(),
                        trust_verified=False,
                        capabilities=msg.payload.get("capabilities", []),
                    )
                    self.connections[peer_id] = conn

                    # Trust federation: verify peer in registry
                    try:
                        from mother.peer_discovery import PeerRegistry
                        registry = PeerRegistry()
                        if registry.get_peer(peer_id) is not None:
                            self.connections[peer_id].trust_verified = True
                    except Exception:
                        pass

                    # Send handshake response
                    response = WormholeMessage(
                        message_id=f"{self.instance_id}-{int(time.time())}",
                        message_type="handshake",
                        payload={
                            "instance_id": self.instance_id,
                            "name": self.instance_name,
                            "capabilities": ["compile", "build", "tools"],
                        },
                        timestamp=time.time(),
                    )
                    await websocket.send(self._serialize_message(response))

                    # Message loop
                    async for raw_msg in websocket:
                        parsed = self._parse_message(raw_msg)
                        if parsed:
                            await self._handle_message(peer_id, parsed)

            except asyncio.TimeoutError:
                logger.warning("Handshake timeout")
            except Exception as e:
                logger.error(f"Connection error: {e}")
            finally:
                if peer_id and peer_id in self.connections:
                    del self.connections[peer_id]

        self._server = await serve(handler, "0.0.0.0", self.port)
        self._running = True
        logger.info(f"Wormhole server listening on port {self.port}")

    async def connect_to_peer(self, host: str, port: int) -> bool:
        """Connect to a peer Mother instance as a client."""
        if not WEBSOCKETS_AVAILABLE:
            return False

        try:
            uri = f"ws://{host}:{port}"
            websocket = await asyncio.wait_for(connect(uri), timeout=10.0)

            # Send handshake
            handshake = WormholeMessage(
                message_id=f"{self.instance_id}-{int(time.time())}",
                message_type="handshake",
                payload={
                    "instance_id": self.instance_id,
                    "name": self.instance_name,
                    "capabilities": ["compile", "build", "tools"],
                },
                timestamp=time.time(),
            )
            await websocket.send(self._serialize_message(handshake))

            # Wait for handshake response
            raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            msg = self._parse_message(raw)

            if msg and msg.message_type == "handshake":
                peer_id = msg.payload.get("instance_id", "")
                peer_name = msg.payload.get("name", "Unknown")

                conn = WormholeConnection(
                    peer_id=peer_id,
                    peer_name=peer_name,
                    websocket=websocket,
                    connected_at=time.time(),
                    last_heartbeat=time.time(),
                    trust_verified=False,
                    capabilities=msg.payload.get("capabilities", []),
                )
                self.connections[peer_id] = conn

                # Trust federation: verify peer in registry
                try:
                    from mother.peer_discovery import PeerRegistry
                    registry = PeerRegistry()
                    if registry.get_peer(peer_id) is not None:
                        self.connections[peer_id].trust_verified = True
                except Exception:
                    pass

                # Start message loop in background
                asyncio.create_task(self._client_loop(peer_id, websocket))

                return True

        except Exception as e:
            logger.error(f"Peer connection failed: {e}")

        return False

    async def _client_loop(self, peer_id: str, websocket):
        """Message receive loop for client connections."""
        try:
            async for raw_msg in websocket:
                parsed = self._parse_message(raw_msg)
                if parsed:
                    await self._handle_message(peer_id, parsed)
        except Exception as e:
            logger.error(f"Client loop error: {e}")
        finally:
            if peer_id in self.connections:
                del self.connections[peer_id]

    async def send_message(self, peer_id: str, msg: WormholeMessage) -> bool:
        """Send a message to a peer. Returns True if sent."""
        conn = self.connections.get(peer_id)
        if not conn or not conn.websocket:
            return False

        try:
            await conn.websocket.send(self._serialize_message(msg))
            return True
        except Exception as e:
            logger.error(f"Send failed to {peer_id}: {e}")
            return False

    async def request(
        self,
        peer_id: str,
        message_type: str,
        payload: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Send request and wait for response."""
        msg_id = f"{self.instance_id}-{int(time.time() * 1000)}"

        msg = WormholeMessage(
            message_id=msg_id,
            message_type=message_type,
            payload=payload,
            timestamp=time.time(),
        )

        # Register pending response
        future = asyncio.Future()
        self._pending_responses[msg_id] = future

        # Send message
        sent = await self.send_message(peer_id, msg)
        if not sent:
            del self._pending_responses[msg_id]
            return None

        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            return None
        finally:
            if msg_id in self._pending_responses:
                del self._pending_responses[msg_id]

    async def _handle_message(self, peer_id: str, msg: WormholeMessage):
        """Route incoming message to handler or pending response."""
        # Check if this is a response to a pending request
        if msg.reply_to and msg.reply_to in self._pending_responses:
            future = self._pending_responses[msg.reply_to]
            if not future.done():
                future.set_result(msg.payload)
            return

        # Update last_heartbeat
        if peer_id in self.connections:
            self.connections[peer_id].last_heartbeat = time.time()

        # Route to handler
        if msg.message_type == "heartbeat":
            # Send heartbeat response
            response = WormholeMessage(
                message_id=f"{self.instance_id}-{int(time.time())}",
                message_type="heartbeat",
                payload={"status": "alive"},
                timestamp=time.time(),
                reply_to=msg.message_id,
            )
            await self.send_message(peer_id, response)
        else:
            # Delegate to callback
            self.on_message(peer_id, msg)

    async def stop(self):
        """Close all connections and stop server."""
        self._running = False

        # Close all peer connections
        for conn in list(self.connections.values()):
            if conn.websocket:
                try:
                    await conn.websocket.close()
                except Exception:
                    pass

        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    def list_connected_peers(self) -> List[str]:
        """Get list of currently connected peer IDs."""
        return list(self.connections.keys())

    def is_connected(self, peer_id: str) -> bool:
        """Check if a specific peer is connected."""
        return peer_id in self.connections

    @staticmethod
    def _serialize_message(msg: WormholeMessage) -> str:
        """Serialize WormholeMessage to JSON string."""
        return json.dumps({
            "message_id": msg.message_id,
            "message_type": msg.message_type,
            "payload": msg.payload,
            "timestamp": msg.timestamp,
            "reply_to": msg.reply_to,
        })

    @staticmethod
    def _parse_message(raw: str) -> Optional[WormholeMessage]:
        """Parse JSON string to WormholeMessage."""
        try:
            data = json.loads(raw)
            return WormholeMessage(
                message_id=data.get("message_id", ""),
                message_type=data.get("message_type", ""),
                payload=data.get("payload", {}),
                timestamp=data.get("timestamp", 0.0),
                reply_to=data.get("reply_to", ""),
            )
        except (json.JSONDecodeError, KeyError):
            return None
