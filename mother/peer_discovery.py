"""
Peer discovery — find other Mother instances on the network.

LEAF module. Uses zeroconf for mDNS discovery + manual peer registration.

Enables Mother to:
- Discover peers via mDNS broadcast
- Register peers manually by IP
- Track peer availability
- Build peer roster for delegation

Optional dependency: zeroconf (graceful degradation if not installed)
"""

import json
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    from zeroconf import ServiceBrowser, ServiceListener, Zeroconf, ServiceInfo
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False
    # Stub types for when zeroconf not installed
    ServiceListener = object
    ServiceBrowser = None
    Zeroconf = None
    ServiceInfo = None


@dataclass(frozen=True)
class PeerInfo:
    """Information about a discovered Mother instance."""

    instance_id: str
    name: str = "Unknown"
    host: str = ""
    port: int = 0
    version: str = ""
    last_seen: float = 0.0
    trust_score: float = 0.0
    capabilities: List[str] = field(default_factory=list)


class PeerRegistry:
    """
    Tracks known Mother instances.

    Stores in ~/.motherlabs/peers.json for persistence across sessions.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (Path.home() / ".motherlabs" / "peers.json")
        self.peers: Dict[str, PeerInfo] = {}
        self._load()

    def _load(self):
        """Load peers from disk."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for p_data in data.get("peers", []):
                    peer = PeerInfo(**p_data)
                    self.peers[peer.instance_id] = peer
            except (json.JSONDecodeError, TypeError):
                pass

    def _save(self):
        """Persist peers to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "peers": [
                {
                    "instance_id": p.instance_id,
                    "name": p.name,
                    "host": p.host,
                    "port": p.port,
                    "version": p.version,
                    "last_seen": p.last_seen,
                    "trust_score": p.trust_score,
                    "capabilities": p.capabilities,
                }
                for p in self.peers.values()
            ]
        }
        self.storage_path.write_text(json.dumps(data, indent=2))

    def add_peer(
        self,
        instance_id: str,
        name: str,
        host: str,
        port: int,
        version: str = "",
        capabilities: Optional[List[str]] = None,
    ) -> PeerInfo:
        """Register a peer. Updates last_seen."""
        peer = PeerInfo(
            instance_id=instance_id,
            name=name,
            host=host,
            port=port,
            version=version,
            last_seen=time.time(),
            capabilities=capabilities or [],
        )
        self.peers[instance_id] = peer
        self._save()
        return peer

    def update_seen(self, instance_id: str) -> None:
        """Update last_seen timestamp for a peer."""
        if instance_id in self.peers:
            old = self.peers[instance_id]
            self.peers[instance_id] = PeerInfo(
                instance_id=old.instance_id,
                name=old.name,
                host=old.host,
                port=old.port,
                version=old.version,
                last_seen=time.time(),
                trust_score=old.trust_score,
                capabilities=old.capabilities,
            )
            self._save()

    def get_peer(self, instance_id: str) -> Optional[PeerInfo]:
        """Get peer by instance_id."""
        return self.peers.get(instance_id)

    def list_peers(self, active_only: bool = False, timeout: float = 300.0) -> List[PeerInfo]:
        """List all peers. Optionally filter to recently seen."""
        peers = list(self.peers.values())
        if active_only:
            cutoff = time.time() - timeout
            peers = [p for p in peers if p.last_seen >= cutoff]
        return sorted(peers, key=lambda p: p.last_seen, reverse=True)

    def update_trust_score(self, instance_id: str, delta: float = 0.1) -> float:
        """Adjust trust score for a peer. Clamps to [0.0, 1.0]. Returns new score."""
        if instance_id not in self.peers:
            return 0.0
        old = self.peers[instance_id]
        new_score = max(0.0, min(1.0, old.trust_score + delta))
        self.peers[instance_id] = PeerInfo(
            instance_id=old.instance_id, name=old.name, host=old.host,
            port=old.port, version=old.version, last_seen=old.last_seen,
            trust_score=new_score, capabilities=old.capabilities,
        )
        self._save()
        return new_score

    def remove_peer(self, instance_id: str) -> bool:
        """Remove a peer from the registry."""
        if instance_id in self.peers:
            del self.peers[instance_id]
            self._save()
            return True
        return False


class MotherServiceListener(ServiceListener):
    """
    Zeroconf service listener for Mother mDNS announcements.

    Listens for _mother._tcp.local. services and registers discovered peers.
    """

    def __init__(self, registry: PeerRegistry):
        self.registry = registry
        self.discovered: Set[str] = set()

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a Mother service is discovered."""
        info = zc.get_service_info(type_, name)
        if not info:
            return

        # Extract instance_id from service name
        # Format: "Mother-{instance_id}._mother._tcp.local."
        if not name.startswith("Mother-"):
            return

        instance_id = name.split("-")[1].split(".")[0]
        if instance_id in self.discovered:
            return

        # Get properties
        properties = {}
        if info.properties:
            for key, value in info.properties.items():
                try:
                    properties[key.decode()] = value.decode()
                except (UnicodeDecodeError, AttributeError):
                    pass

        # Extract host
        addresses = info.parsed_addresses()
        host = addresses[0] if addresses else ""
        port = info.port

        self.registry.add_peer(
            instance_id=instance_id,
            name=properties.get("name", "Mother"),
            host=host,
            port=port,
            version=properties.get("version", ""),
            capabilities=properties.get("capabilities", "").split(",") if properties.get("capabilities") else [],
        )
        self.discovered.add(instance_id)

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service is updated."""
        if name.startswith("Mother-"):
            instance_id = name.split("-")[1].split(".")[0]
            self.registry.update_seen(instance_id)

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when a service disappears."""
        pass  # Peers remain in registry, just not updated


def start_discovery(registry: PeerRegistry, timeout: float = 5.0) -> int:
    """
    Start mDNS discovery for Mother instances.

    Runs for timeout seconds, returns number of peers discovered.
    Requires zeroconf library.
    """
    if not ZEROCONF_AVAILABLE:
        return 0

    try:
        zc = Zeroconf()
        listener = MotherServiceListener(registry)
        browser = ServiceBrowser(zc, "_mother._tcp.local.", listener)

        time.sleep(timeout)

        browser.cancel()
        zc.close()

        return len(listener.discovered)
    except Exception:
        return 0


def announce_self(
    instance_id: str,
    name: str,
    port: int,
    version: str = "",
    capabilities: Optional[List[str]] = None,
) -> Optional[ServiceInfo]:
    """
    Announce this Mother instance via mDNS.

    Returns ServiceInfo if successful, None if zeroconf unavailable.
    Keep the returned ServiceInfo alive to maintain the announcement.
    """
    if not ZEROCONF_AVAILABLE:
        return None

    try:
        # Get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()

        service_name = f"Mother-{instance_id}._mother._tcp.local."

        properties = {
            "name": name,
            "version": version,
        }
        if capabilities:
            properties["capabilities"] = ",".join(capabilities)

        info = ServiceInfo(
            "_mother._tcp.local.",
            service_name,
            addresses=[socket.inet_aton(local_ip)],
            port=port,
            properties=properties,
        )

        zc = Zeroconf()
        zc.register_service(info)

        return info

    except Exception:
        return None
