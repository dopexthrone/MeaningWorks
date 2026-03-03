"""
kernel/provenance_signing.py — Cryptographic provenance signing.

LEAF module. Stdlib only (hmac, hashlib, json, os).
No imports from kernel/, core/, or mother/.

Signs provenance chains with HMAC-SHA256 so forged source tuples
can be detected. Each instance generates a persistent signing key
on first use (stored at ~/.motherlabs/provenance.key).

Usage:
    signer = ProvenanceSigner()  # loads or creates key
    sig = signer.sign_provenance(source=("human:test", "__intent_contract__"), postcode="INT.ENT.ECO.WHAT.SFT")
    assert signer.verify(source, postcode, sig)

The signature covers: sorted source tuple + postcode + fill state.
This means any mutation to the provenance chain invalidates the signature.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


_DEFAULT_KEY_DIR = Path.home() / ".motherlabs"
_DEFAULT_KEY_FILE = "provenance.key"
_KEY_BYTES = 32  # 256-bit key


@dataclass(frozen=True)
class ProvenanceSignature:
    """A signed provenance assertion."""
    signature: str          # hex-encoded HMAC-SHA256
    source_hash: str        # SHA256 of the signed content (for debug/audit)
    signer_id: str          # first 8 chars of key fingerprint


def _key_path(key_dir: Optional[Path] = None) -> Path:
    d = key_dir or _DEFAULT_KEY_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / _DEFAULT_KEY_FILE


def _load_or_create_key(key_dir: Optional[Path] = None) -> bytes:
    """Load signing key from disk, or create if missing."""
    path = _key_path(key_dir)
    if path.exists():
        raw = path.read_bytes()
        if len(raw) >= _KEY_BYTES:
            return raw[:_KEY_BYTES]
    # Generate new key
    key = secrets.token_bytes(_KEY_BYTES)
    path.write_bytes(key)
    # Restrict permissions (owner-only read/write)
    try:
        os.chmod(str(path), 0o600)
    except OSError:
        pass  # Windows or restricted filesystem
    return key


def _key_fingerprint(key: bytes) -> str:
    """First 8 hex chars of SHA256 of the key — safe to share."""
    return hashlib.sha256(key).hexdigest()[:8]


def _canonical_message(
    source: Tuple[str, ...],
    postcode: str,
    fill_state: str = "",
    content_hash: str = "",
) -> str:
    """Build the canonical message to sign.

    Deterministic: sorted source, lowercase postcode.
    """
    parts = {
        "source": sorted(source),
        "postcode": postcode.strip().upper(),
    }
    if fill_state:
        parts["fill"] = fill_state.strip().upper()
    if content_hash:
        parts["content_hash"] = content_hash
    return json.dumps(parts, sort_keys=True, separators=(",", ":"))


class ProvenanceSigner:
    """HMAC-SHA256 signer for provenance chains."""

    def __init__(self, key_dir: Optional[Path] = None):
        self._key = _load_or_create_key(key_dir)
        self._fingerprint = _key_fingerprint(self._key)

    @property
    def signer_id(self) -> str:
        """Public fingerprint of the signing key."""
        return self._fingerprint

    def sign_provenance(
        self,
        source: Tuple[str, ...],
        postcode: str,
        fill_state: str = "",
        content_hash: str = "",
    ) -> ProvenanceSignature:
        """Sign a provenance assertion.

        Args:
            source: the provenance source tuple (e.g., ("human:test",))
            postcode: the cell postcode key
            fill_state: optional fill state (F, P, C, etc.)
            content_hash: optional SHA256 of cell content

        Returns:
            ProvenanceSignature with HMAC signature
        """
        msg = _canonical_message(source, postcode, fill_state, content_hash)
        msg_bytes = msg.encode("utf-8")

        sig = hmac.new(self._key, msg_bytes, hashlib.sha256).hexdigest()
        source_hash = hashlib.sha256(msg_bytes).hexdigest()[:16]

        return ProvenanceSignature(
            signature=sig,
            source_hash=source_hash,
            signer_id=self._fingerprint,
        )

    def verify(
        self,
        source: Tuple[str, ...],
        postcode: str,
        signature: ProvenanceSignature,
        fill_state: str = "",
        content_hash: str = "",
    ) -> bool:
        """Verify a provenance signature.

        Returns True if the signature is valid for the given provenance data.
        """
        if signature.signer_id != self._fingerprint:
            return False  # signed by a different key

        msg = _canonical_message(source, postcode, fill_state, content_hash)
        msg_bytes = msg.encode("utf-8")

        expected = hmac.new(self._key, msg_bytes, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature.signature)

    def sign_chain(
        self,
        chain: Tuple[Tuple[str, ...], ...],
        postcodes: Tuple[str, ...],
    ) -> ProvenanceSignature:
        """Sign an entire provenance chain (multiple cells).

        Useful for signing a path from intent → leaf cell.
        """
        combined = json.dumps(
            {"chain": [sorted(s) for s in chain], "postcodes": list(postcodes)},
            sort_keys=True,
            separators=(",", ":"),
        )
        msg_bytes = combined.encode("utf-8")

        sig = hmac.new(self._key, msg_bytes, hashlib.sha256).hexdigest()
        source_hash = hashlib.sha256(msg_bytes).hexdigest()[:16]

        return ProvenanceSignature(
            signature=sig,
            source_hash=source_hash,
            signer_id=self._fingerprint,
        )

    def verify_chain(
        self,
        chain: Tuple[Tuple[str, ...], ...],
        postcodes: Tuple[str, ...],
        signature: ProvenanceSignature,
    ) -> bool:
        """Verify a chain signature."""
        if signature.signer_id != self._fingerprint:
            return False

        combined = json.dumps(
            {"chain": [sorted(s) for s in chain], "postcodes": list(postcodes)},
            sort_keys=True,
            separators=(",", ":"),
        )
        msg_bytes = combined.encode("utf-8")
        expected = hmac.new(self._key, msg_bytes, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature.signature)


def content_hash(content: str) -> str:
    """SHA256 hash of cell content for signing."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
