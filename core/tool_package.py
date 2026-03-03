"""
Motherlabs Tool Package — the atomic unit of cross-instance tool sharing.

LEAF MODULE — stdlib only (hashlib, json, time). Zero project imports.

Provides:
- ProvenanceRecord: Frozen provenance chain entry
- ToolPackage: Complete tool bundle (blueprint + code + trust + provenance)
- ToolDigest: Lightweight summary for discovery (no code)
- Pure functions for packaging, serialization, and ID computation
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class ProvenanceRecord:
    """A single entry in the provenance chain.

    Records which instance compiled what, when, and with what trust level.
    """
    compilation_id: str        # Hash of the original compilation
    instance_id: str           # Which instance compiled this
    timestamp: str             # ISO8601
    input_hash: str            # SHA256[:16] of original input text
    domain: str                # Domain adapter name
    provider: str              # LLM provider used
    trust_score: float         # Overall trust score at compilation time
    verification_badge: str    # "verified" | "partial" | "unverified"


@dataclass(frozen=True)
class ToolPackage:
    """Complete tool bundle for cross-instance sharing.

    Contains everything needed to validate and reuse a compiled tool:
    blueprint, generated code, trust indicators, and full provenance chain.
    """
    package_id: str                          # SHA256[:16] of canonical content
    name: str                                # Human-readable tool name
    version: str                             # Semantic version (1.0.0)
    created_at: str                          # ISO8601
    blueprint: Dict[str, Any]
    generated_code: Dict[str, str]           # Component name -> code
    file_extension: str                      # .py, .yaml
    domain: str
    trust_score: float                       # 0-100
    verification_badge: str
    fidelity_scores: Dict[str, int]          # 7 verification dimensions
    fingerprint: str                         # StructuralFingerprint.hash_digest
    provenance_chain: Tuple[ProvenanceRecord, ...]
    compilation_id: str
    source_instance_id: str
    usage_count: int = 0


@dataclass(frozen=True)
class ToolDigest:
    """Lightweight summary for discovery (no code).

    Used in list/search results to avoid transferring full code.
    """
    package_id: str
    name: str
    domain: str
    fingerprint: str
    trust_score: float
    verification_badge: str
    component_count: int
    relationship_count: int
    source_instance_id: str
    created_at: str


# =============================================================================
# PACKAGE ID COMPUTATION
# =============================================================================

def compute_package_id(blueprint: Dict[str, Any], generated_code: Dict[str, str]) -> str:
    """Compute deterministic package ID from canonical content.

    The ID is SHA256[:16] of the canonical JSON representation of the
    blueprint topology and generated code. Same content = same ID.

    Args:
        blueprint: The tool's blueprint dict
        generated_code: Component name -> code mapping

    Returns:
        16-character hex string
    """
    # Build canonical structure — sorted keys for determinism
    components = blueprint.get("components", [])
    relationships = blueprint.get("relationships", [])

    canonical = {
        "components": sorted(
            [c.get("name", "") for c in components]
        ),
        "relationships": sorted(
            [(r.get("from", ""), r.get("to", ""), r.get("type", ""))
             for r in relationships]
        ),
        "code": {k: v for k, v in sorted(generated_code.items())},
    }

    content = json.dumps(canonical, sort_keys=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


# =============================================================================
# PACKAGING
# =============================================================================

def package_tool(
    compilation_id: str,
    blueprint: Dict[str, Any],
    generated_code: Dict[str, str],
    trust_score: float,
    verification_badge: str,
    fidelity_scores: Dict[str, int],
    fingerprint_hash: str,
    instance_id: str,
    domain: str = "software",
    file_extension: str = ".py",
    provider: str = "unknown",
    input_hash: str = "",
    name: Optional[str] = None,
    version: str = "1.0.0",
) -> ToolPackage:
    """Create a ToolPackage from compilation outputs.

    Args:
        compilation_id: Hash identifying the compilation
        blueprint: Compiled blueprint dict
        generated_code: Component name -> code mapping
        trust_score: Overall trust score (0-100)
        verification_badge: "verified" | "partial" | "unverified"
        fidelity_scores: 7 verification dimensions -> scores
        fingerprint_hash: StructuralFingerprint.hash_digest
        instance_id: This instance's ID
        domain: Domain adapter name
        file_extension: Output file extension
        provider: LLM provider used
        input_hash: SHA256[:16] of original input
        name: Human-readable name (defaults to blueprint core_need)
        version: Semantic version string

    Returns:
        Frozen ToolPackage
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    package_id = compute_package_id(blueprint, generated_code)

    # Derive name from blueprint if not provided
    if not name:
        name = blueprint.get("core_need", "unnamed-tool")
        # Truncate long names
        if len(name) > 80:
            name = name[:77] + "..."

    # Build initial provenance record
    provenance = ProvenanceRecord(
        compilation_id=compilation_id,
        instance_id=instance_id,
        timestamp=now,
        input_hash=input_hash,
        domain=domain,
        provider=provider,
        trust_score=trust_score,
        verification_badge=verification_badge,
    )

    return ToolPackage(
        package_id=package_id,
        name=name,
        version=version,
        created_at=now,
        blueprint=blueprint,
        generated_code=generated_code,
        file_extension=file_extension,
        domain=domain,
        trust_score=trust_score,
        verification_badge=verification_badge,
        fidelity_scores=fidelity_scores,
        fingerprint=fingerprint_hash,
        provenance_chain=(provenance,),
        compilation_id=compilation_id,
        source_instance_id=instance_id,
        usage_count=0,
    )


# =============================================================================
# DIGEST EXTRACTION
# =============================================================================

def extract_digest(pkg: ToolPackage) -> ToolDigest:
    """Extract lightweight digest from a full ToolPackage.

    Args:
        pkg: Full ToolPackage

    Returns:
        ToolDigest (no code, no blueprint)
    """
    components = pkg.blueprint.get("components", [])
    relationships = pkg.blueprint.get("relationships", [])

    return ToolDigest(
        package_id=pkg.package_id,
        name=pkg.name,
        domain=pkg.domain,
        fingerprint=pkg.fingerprint,
        trust_score=pkg.trust_score,
        verification_badge=pkg.verification_badge,
        component_count=len(components),
        relationship_count=len(relationships),
        source_instance_id=pkg.source_instance_id,
        created_at=pkg.created_at,
    )


# =============================================================================
# SERIALIZATION
# =============================================================================

def _serialize_provenance_record(rec: ProvenanceRecord) -> Dict[str, Any]:
    """Serialize a ProvenanceRecord to JSON-safe dict."""
    return {
        "compilation_id": rec.compilation_id,
        "instance_id": rec.instance_id,
        "timestamp": rec.timestamp,
        "input_hash": rec.input_hash,
        "domain": rec.domain,
        "provider": rec.provider,
        "trust_score": rec.trust_score,
        "verification_badge": rec.verification_badge,
    }


def _deserialize_provenance_record(data: Dict[str, Any]) -> ProvenanceRecord:
    """Deserialize a ProvenanceRecord from dict."""
    return ProvenanceRecord(
        compilation_id=data["compilation_id"],
        instance_id=data["instance_id"],
        timestamp=data["timestamp"],
        input_hash=data.get("input_hash", ""),
        domain=data.get("domain", "software"),
        provider=data.get("provider", "unknown"),
        trust_score=float(data.get("trust_score", 0.0)),
        verification_badge=data.get("verification_badge", "unverified"),
    )


def serialize_tool_package(pkg: ToolPackage) -> Dict[str, Any]:
    """Serialize a ToolPackage to JSON-safe dict.

    This is the .mtool file format.

    Args:
        pkg: ToolPackage to serialize

    Returns:
        JSON-safe dict
    """
    return {
        "format": "mtool",
        "format_version": "1.0",
        "package_id": pkg.package_id,
        "name": pkg.name,
        "version": pkg.version,
        "created_at": pkg.created_at,
        "blueprint": pkg.blueprint,
        "generated_code": pkg.generated_code,
        "file_extension": pkg.file_extension,
        "domain": pkg.domain,
        "trust_score": pkg.trust_score,
        "verification_badge": pkg.verification_badge,
        "fidelity_scores": pkg.fidelity_scores,
        "fingerprint": pkg.fingerprint,
        "provenance_chain": [
            _serialize_provenance_record(rec)
            for rec in pkg.provenance_chain
        ],
        "compilation_id": pkg.compilation_id,
        "source_instance_id": pkg.source_instance_id,
        "usage_count": pkg.usage_count,
    }


def deserialize_tool_package(data: Dict[str, Any]) -> ToolPackage:
    """Deserialize a ToolPackage from dict.

    Args:
        data: Dict (e.g. from .mtool JSON file)

    Returns:
        Frozen ToolPackage

    Raises:
        KeyError: If required fields are missing
        ValueError: If data is malformed
    """
    provenance_chain = tuple(
        _deserialize_provenance_record(rec)
        for rec in data.get("provenance_chain", [])
    )

    return ToolPackage(
        package_id=data["package_id"],
        name=data["name"],
        version=data.get("version", "1.0.0"),
        created_at=data["created_at"],
        blueprint=data["blueprint"],
        generated_code=data["generated_code"],
        file_extension=data.get("file_extension", ".py"),
        domain=data.get("domain", "software"),
        trust_score=float(data.get("trust_score", 0.0)),
        verification_badge=data.get("verification_badge", "unverified"),
        fidelity_scores=data.get("fidelity_scores", {}),
        fingerprint=data.get("fingerprint", ""),
        provenance_chain=provenance_chain,
        compilation_id=data.get("compilation_id", ""),
        source_instance_id=data.get("source_instance_id", ""),
        usage_count=int(data.get("usage_count", 0)),
    )


def serialize_digest(digest: ToolDigest) -> Dict[str, Any]:
    """Serialize a ToolDigest to JSON-safe dict.

    Args:
        digest: ToolDigest to serialize

    Returns:
        JSON-safe dict
    """
    return {
        "package_id": digest.package_id,
        "name": digest.name,
        "domain": digest.domain,
        "fingerprint": digest.fingerprint,
        "trust_score": digest.trust_score,
        "verification_badge": digest.verification_badge,
        "component_count": digest.component_count,
        "relationship_count": digest.relationship_count,
        "source_instance_id": digest.source_instance_id,
        "created_at": digest.created_at,
    }


def deserialize_digest(data: Dict[str, Any]) -> ToolDigest:
    """Deserialize a ToolDigest from dict.

    Args:
        data: Dict

    Returns:
        Frozen ToolDigest
    """
    return ToolDigest(
        package_id=data["package_id"],
        name=data["name"],
        domain=data.get("domain", "software"),
        fingerprint=data.get("fingerprint", ""),
        trust_score=float(data.get("trust_score", 0.0)),
        verification_badge=data.get("verification_badge", "unverified"),
        component_count=int(data.get("component_count", 0)),
        relationship_count=int(data.get("relationship_count", 0)),
        source_instance_id=data.get("source_instance_id", ""),
        created_at=data.get("created_at", ""),
    )
