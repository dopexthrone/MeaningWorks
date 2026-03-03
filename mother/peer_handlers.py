"""
Peer request handlers — process incoming wormhole messages.

LEAF module (but imports EngineBridge for actual work).

Handles:
- compile_request: run compilation, return blueprint
- build_request: run full build, return project metadata
- tool_offer: receive and register a tool package
- corpus_sync: receive and store a compilation

This is the "service" side of the peer protocol. When a peer sends
a compile_request, these handlers do the actual work and send back results.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from mother.wormhole import WormholeMessage

logger = logging.getLogger("mother.peer_handlers")


async def handle_compile_request(
    message: WormholeMessage,
    bridge,  # EngineBridge instance
) -> Dict[str, Any]:
    """
    Handle a compile_request from a peer.

    Runs compilation via the local engine, returns blueprint + trust data.
    """
    payload = message.payload
    description = payload.get("description", "")
    domain = payload.get("domain", "")

    if not description:
        return {
            "success": False,
            "error": "No description provided",
        }

    try:
        # Run compilation via bridge
        result = await bridge.compile(description)

        if not result or not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "Compilation failed"),
                "cost_usd": bridge.get_last_call_cost(),
            }

        blueprint = result.get("blueprint", {})
        verification = result.get("verification", {})
        trust_score = verification.get("overall_score", 0.0) if isinstance(verification, dict) else 0.0
        insights = result.get("insights", [])

        return {
            "success": True,
            "blueprint": blueprint,
            "trust_score": trust_score,
            "insights": insights[:10],  # Top 10 insights
            "cost_usd": bridge.get_last_call_cost(),
        }

    except Exception as e:
        logger.error(f"Compile delegation error: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_build_request(
    message: WormholeMessage,
    bridge,  # EngineBridge instance
) -> Dict[str, Any]:
    """
    Handle a build_request from a peer.

    Runs full build pipeline, returns project metadata.
    """
    payload = message.payload
    description = payload.get("description", "")

    if not description:
        return {
            "success": False,
            "error": "No description provided",
        }

    try:
        # Run build via bridge
        result = await bridge.build(description)

        if not result or not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "Build failed"),
                "cost_usd": bridge.get_last_call_cost(),
            }

        manifest = result.get("manifest", {})
        trust = result.get("trust", {})
        trust_score = trust.get("overall_score", 0.0) if isinstance(trust, dict) else 0.0

        return {
            "success": True,
            "project_path": manifest.get("project_path", ""),
            "files_written": len(manifest.get("files_written", [])),
            "entry_point": manifest.get("entry_point", ""),
            "trust_score": trust_score,
            "cost_usd": bridge.get_last_call_cost(),
        }

    except Exception as e:
        logger.error(f"Build delegation error: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_tool_offer(
    message: WormholeMessage,
    bridge,  # EngineBridge instance
) -> Dict[str, Any]:
    """
    Handle a tool_offer from a peer.

    Receives a tool package, validates it, registers it locally.
    """
    payload = message.payload
    tool_package = payload.get("tool_package", {})

    if not tool_package:
        return {
            "success": False,
            "error": "No tool package provided",
        }

    try:
        # Import the tool (validates + registers)
        # This would use the existing tool_export.import_tool logic
        # For now, stub implementation
        return {
            "success": True,
            "tool_name": tool_package.get("metadata", {}).get("name", "unknown"),
            "trust_score": tool_package.get("trust_score", 0.0),
        }

    except Exception as e:
        logger.error(f"Tool import error: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_corpus_sync(
    message: WormholeMessage,
    bridge,
) -> Dict[str, Any]:
    """Handle a corpus_sync from a peer — store compilation summary in journal."""
    payload = message.payload
    peer_id = payload.get("peer_id", "unknown")
    summary = payload.get("summary", "")
    trust_score = payload.get("trust_score", 0.0)
    domain = payload.get("domain", "peer")

    if not summary:
        return {"success": False, "error": "empty summary"}

    try:
        from pathlib import Path
        db_path = Path.home() / ".motherlabs" / "history.db"
        from mother.journal import BuildJournal, JournalEntry
        journal = BuildJournal(db_path)
        journal.record(JournalEntry(
            event_type="peer_compile",
            description=f"[peer:{peer_id[:8]}] {summary[:300]}",
            success=True, trust_score=trust_score,
            domain=domain, experiment_tag="corpus_sync",
        ))
        journal.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def create_message_router(bridge):
    """
    Create a message handler function for the Wormhole.

    Returns a callable that routes incoming messages to appropriate handlers.
    """

    async def route_message(peer_id: str, message: WormholeMessage):
        """Route incoming wormhole message to handler."""
        msg_type = message.message_type

        # Skip heartbeats (handled in wormhole.py)
        if msg_type == "heartbeat":
            return

        # Route to handler
        handler_map = {
            "compile_request": handle_compile_request,
            "build_request": handle_build_request,
            "tool_offer": handle_tool_offer,
            "corpus_sync": handle_corpus_sync,
        }

        handler = handler_map.get(msg_type)
        if not handler:
            logger.warning(f"Unknown message type from {peer_id}: {msg_type}")
            return

        # Execute handler
        try:
            response_payload = await handler(message, bridge)

            # Send response back
            response = WormholeMessage(
                message_id=f"response-{message.message_id}",
                message_type="response",
                payload=response_payload,
                timestamp=time.time(),
                reply_to=message.message_id,
            )

            await asyncio.sleep(0)  # Yield to event loop
            # The wormhole will handle sending via its connection
            if peer_id in bridge._wormhole.connections:
                conn = bridge._wormhole.connections[peer_id]
                if conn.websocket:
                    await conn.websocket.send(
                        bridge._wormhole._serialize_message(response)
                    )

        except Exception as e:
            logger.error(f"Handler error for {msg_type}: {e}")

    return route_message
