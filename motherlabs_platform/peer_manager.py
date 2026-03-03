"""
Peer manager — orchestrates peer lifecycle, heartbeat, and tool sync.

Imports from instance_identity + peer_client.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from motherlabs_platform.instance_identity import InstanceIdentityStore, InstanceRecord
from motherlabs_platform.peer_client import PeerClient, PeerStatus

logger = logging.getLogger(__name__)


@dataclass
class PeerState:
    """Mutable state for a connected peer."""
    record: InstanceRecord
    client: PeerClient
    status: PeerStatus
    last_heartbeat: float = 0.0
    consecutive_failures: int = 0
    is_online: bool = False


class PeerManager:
    """Orchestrates peer discovery, heartbeat, and tool exchange.

    Lifecycle: start() → add_peer() / sync / heartbeat → stop()
    """

    def __init__(
        self,
        identity_store: InstanceIdentityStore,
        heartbeat_interval: float = 60.0,
        max_failures: int = 3,
    ):
        self._store = identity_store
        self._heartbeat_interval = heartbeat_interval
        self._max_failures = max_failures
        self._peers: Dict[str, PeerState] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start the peer manager and heartbeat loop."""
        if self._running:
            return
        self._running = True

        # Load known peers from identity store
        for record in self._store.list_peers():
            if record.instance_id not in self._peers:
                client = PeerClient(record.api_endpoint)
                now = time.strftime("%Y-%m-%dT%H:%M:%S")
                self._peers[record.instance_id] = PeerState(
                    record=record,
                    client=client,
                    status=PeerStatus(
                        instance_id=record.instance_id,
                        reachable=False,
                        latency_ms=0.0,
                        last_checked=now,
                    ),
                )

        # Initial health check for all peers
        await self._check_all_peers()

        # Start heartbeat loop
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(f"PeerManager started with {len(self._peers)} known peers")

    async def stop(self) -> None:
        """Stop the heartbeat loop and close all peer clients."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Close all clients
        for state in self._peers.values():
            try:
                await state.client.close()
            except Exception:
                pass
        logger.info("PeerManager stopped")

    async def add_peer(
        self,
        instance_id: str,
        name: str,
        endpoint: str,
    ) -> PeerStatus:
        """Register a new peer, attempt mutual registration and health check.

        Returns the peer's health status.
        """
        # Register in local store
        self._store.register_peer(instance_id, name, endpoint)
        record = self._store.get_peer(instance_id)
        if not record:
            record = InstanceRecord(
                instance_id=instance_id,
                name=name,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                api_endpoint=endpoint,
                is_self=False,
            )

        client = PeerClient(endpoint)

        # Health check
        status = await client.health_check(instance_id)

        state = PeerState(
            record=record,
            client=client,
            status=status,
            last_heartbeat=time.monotonic(),
            is_online=status.reachable,
        )
        self._peers[instance_id] = state

        # Mutual registration: tell the remote peer about us
        if status.reachable:
            self_record = self._store.get_or_create_self()
            if self_record:
                await client.register_self(
                    instance_id=self_record.instance_id,
                    name=self_record.name,
                    api_endpoint=self_record.api_endpoint,
                )

        return status

    async def remove_peer(self, instance_id: str) -> bool:
        """Remove a peer from the manager and identity store."""
        state = self._peers.pop(instance_id, None)
        if state:
            try:
                await state.client.close()
            except Exception:
                pass
        return self._store.remove_peer(instance_id)

    def get_online_peers(self) -> List[PeerState]:
        """Return all currently online peers."""
        return [s for s in self._peers.values() if s.is_online]

    def get_all_peers(self) -> List[PeerState]:
        """Return all known peers."""
        return list(self._peers.values())

    async def sync_tool_digests(self, instance_id: str) -> List[Dict[str, Any]]:
        """Exchange tool digests with a peer. Returns list of new tools found.

        Compares remote digest with local tools to find unknown entries.
        """
        state = self._peers.get(instance_id)
        if not state or not state.is_online:
            return []

        try:
            remote_tools = await state.client.list_tools()
            # Compare with local registry
            from motherlabs_platform.tool_registry import get_tool_registry
            registry = get_tool_registry()
            local_tools = registry.list_tools()
            local_ids = {t.package_id for t in local_tools}

            new_tools = [
                t for t in remote_tools
                if t.get("package_id") not in local_ids
            ]
            return new_tools
        except Exception as e:
            logger.warning(f"Tool digest sync failed for {instance_id}: {e}")
            return []

    async def pull_tool(
        self,
        instance_id: str,
        package_id: str,
    ) -> Dict[str, Any]:
        """Fetch a tool from a peer and import it locally.

        Returns dict with import result.
        """
        state = self._peers.get(instance_id)
        if not state or not state.is_online:
            return {"success": False, "error": "Peer not online"}

        try:
            tool_data = await state.client.get_tool(package_id)

            # Import via tool registry
            from motherlabs_platform.tool_registry import get_tool_registry
            registry = get_tool_registry()

            # Use the V2 import mechanism
            from core.tool_package import ToolPackage
            pkg = ToolPackage(
                package_id=tool_data.get("package_id", package_id),
                name=tool_data.get("name", "unknown"),
                version=tool_data.get("version", "0.1.0"),
                domain=tool_data.get("domain", "software"),
                trust_score=tool_data.get("trust_score", 0.0),
                verification_badge=tool_data.get("verification_badge", "unverified"),
                fingerprint=tool_data.get("fingerprint", ""),
                blueprint=tool_data.get("blueprint", {}),
                generated_code=tool_data.get("generated_code", {}),
                fidelity_scores=tool_data.get("fidelity_scores", {}),
                provenance_chain=tool_data.get("provenance_chain", []),
                source_instance_id=tool_data.get("source_instance_id", instance_id),
            )

            # Governor validation: check trust score
            if pkg.trust_score < 40.0:
                return {
                    "success": False,
                    "error": f"Trust score too low ({pkg.trust_score:.0f}%)",
                    "package_id": package_id,
                }

            try:
                registry.register_tool(pkg, is_local=False)
            except Exception as e:
                return {"success": False, "error": str(e), "package_id": package_id}

            return {
                "success": True,
                "package_id": pkg.package_id,
                "name": pkg.name,
                "trust_score": pkg.trust_score,
            }

        except Exception as e:
            logger.warning(f"Tool pull failed for {package_id} from {instance_id}: {e}")
            return {"success": False, "error": str(e), "package_id": package_id}

    async def _heartbeat_loop(self) -> None:
        """Periodic health check for all peers."""
        while self._running:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                if not self._running:
                    break
                await self._check_all_peers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")

    async def _check_all_peers(self) -> None:
        """Check health of all known peers."""
        for instance_id, state in list(self._peers.items()):
            try:
                status = await state.client.health_check(instance_id)
                state.status = status
                state.last_heartbeat = time.monotonic()

                if status.reachable:
                    state.consecutive_failures = 0
                    state.is_online = True
                else:
                    state.consecutive_failures += 1
                    if state.consecutive_failures >= self._max_failures:
                        state.is_online = False
                        logger.info(
                            f"Peer {instance_id} marked offline after "
                            f"{state.consecutive_failures} failures"
                        )
            except Exception as e:
                state.consecutive_failures += 1
                if state.consecutive_failures >= self._max_failures:
                    state.is_online = False
                logger.debug(f"Health check failed for {instance_id}: {e}")
