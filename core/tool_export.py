"""
Motherlabs Tool Export/Import — file-based tool sharing.

Near-leaf module — imports from core.tool_package, core.governor_validation.

Provides:
- export_tool(): Package a compilation from corpus into a ToolPackage
- export_tool_to_file(): Write a ToolPackage to a .mtool JSON file
- load_tool_from_file(): Read a ToolPackage from a .mtool JSON file
- import_tool(): Import a validated tool into the local registry
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("motherlabs.tool_export")

from core.tool_package import (
    ToolPackage,
    compute_package_id,
    deserialize_tool_package,
    package_tool,
    serialize_tool_package,
)
from core.governor_validation import (
    ImportValidationResult,
    validate_import,
)


# =============================================================================
# EXPORT
# =============================================================================

def export_tool(
    compilation_id: str,
    corpus,
    instance_id: str,
    name: Optional[str] = None,
    version: str = "1.0.0",
) -> ToolPackage:
    """Package a compilation from corpus into a ToolPackage.

    Loads blueprint, context_graph, and generated_code from corpus,
    computes fingerprint and trust indicators, builds provenance chain.

    Args:
        compilation_id: Corpus compilation ID (hash)
        corpus: Corpus instance (persistence.corpus.Corpus)
        instance_id: This instance's ID
        name: Human-readable name (defaults to blueprint core_need)
        version: Semantic version string

    Returns:
        Frozen ToolPackage

    Raises:
        ValueError: If compilation not found or not successful
    """
    # Load from corpus
    record = corpus.get(compilation_id)
    if not record:
        raise ValueError(f"Compilation not found: {compilation_id}")

    blueprint = corpus.load_blueprint(compilation_id)
    if not blueprint:
        raise ValueError(f"Blueprint not found for: {compilation_id}")

    context_graph = corpus.load_context_graph(compilation_id) or {}

    # Extract generated_code from context_graph (stored after emission)
    generated_code = context_graph.get("generated_code", {})
    if not generated_code:
        # Try emission_result
        emission = context_graph.get("emission_result", {})
        if isinstance(emission, dict):
            generated_code = emission.get("generated_code", {})

    # Compute fingerprint
    from core.determinism import compute_structural_fingerprint
    fingerprint = compute_structural_fingerprint(blueprint)

    # Compute trust indicators
    from core.trust import compute_trust_indicators, serialize_trust_indicators
    dim_meta = context_graph.get("dimensional_metadata", {})
    verification = context_graph.get("verification", {})
    intent_keywords = list(context_graph.get("keywords", []))

    trust = compute_trust_indicators(
        blueprint=blueprint,
        verification=verification,
        context_graph=context_graph,
        dimensional_metadata=dim_meta,
        intent_keywords=intent_keywords,
    )

    # Compute input hash
    input_text = getattr(record, "input_text", "") or ""
    input_hash = hashlib.sha256(input_text.encode("utf-8")).hexdigest()[:16]

    # Build domain and provider from record
    domain = getattr(record, "domain", "software") or "software"
    provider = getattr(record, "provider", "unknown") or "unknown"

    return package_tool(
        compilation_id=compilation_id,
        blueprint=blueprint,
        generated_code=generated_code,
        trust_score=trust.overall_score,
        verification_badge=trust.verification_badge,
        fidelity_scores=dict(trust.fidelity_scores),
        fingerprint_hash=fingerprint.hash_digest,
        instance_id=instance_id,
        domain=domain,
        file_extension=".py",
        provider=provider,
        input_hash=input_hash,
        name=name,
        version=version,
    )


# =============================================================================
# FILE I/O
# =============================================================================

def export_tool_to_file(
    package: ToolPackage,
    output_path: Optional[str] = None,
) -> str:
    """Write a ToolPackage to a .mtool JSON file.

    Args:
        package: ToolPackage to export
        output_path: File path (defaults to ./{name}.mtool)

    Returns:
        Absolute path of written file
    """
    if not output_path:
        # Sanitize name for filename
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in package.name.lower()
        ).strip("_")[:60]
        output_path = f"./{safe_name}.mtool"

    # Ensure parent directory exists
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = serialize_tool_package(package)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return str(path.resolve())


def load_tool_from_file(file_path: str) -> ToolPackage:
    """Read a ToolPackage from a .mtool JSON file.

    Args:
        file_path: Path to .mtool file

    Returns:
        Frozen ToolPackage

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid .mtool JSON
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Tool file not found: {file_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tool file: {e}")

    # Validate format marker
    if data.get("format") != "mtool":
        raise ValueError(
            f"Not a .mtool file: format={data.get('format', 'missing')}"
        )

    try:
        return deserialize_tool_package(data)
    except (KeyError, TypeError) as e:
        raise ValueError(f"Malformed .mtool file: {e}")


# =============================================================================
# IMPORT
# =============================================================================

def import_tool(
    package: ToolPackage,
    registry,
    instance_id: str,
    min_trust_score: float = 60.0,
    require_verified: bool = False,
) -> ImportValidationResult:
    """Import a tool package into the local registry after governor validation.

    Runs full governor validation. If validation passes, registers in registry.
    Returns False (via result.allowed) if duplicate fingerprint already exists.

    Args:
        package: ToolPackage to import
        registry: ToolRegistry instance
        instance_id: This instance's ID
        min_trust_score: Minimum trust score for import
        require_verified: If True, require "verified" badge

    Returns:
        ImportValidationResult (check .allowed for success)
    """
    # Check for duplicate fingerprint
    existing = registry.find_by_fingerprint(package.fingerprint)
    if existing:
        return ImportValidationResult(
            allowed=False,
            rejection_reason=f"Duplicate fingerprint: tool '{existing.name}' already registered",
            provenance_valid=True,
            trust_sufficient=True,
            code_safe=True,
            warnings=(),
            provenance_depth=len(package.provenance_chain),
            trust_score=package.trust_score,
            checks_performed=("fingerprint_dedup",),
        )

    # Serialize provenance chain for validation
    provenance_dicts = [
        {
            "compilation_id": rec.compilation_id,
            "instance_id": rec.instance_id,
            "timestamp": rec.timestamp,
            "input_hash": rec.input_hash,
            "domain": rec.domain,
            "provider": rec.provider,
            "trust_score": rec.trust_score,
            "verification_badge": rec.verification_badge,
        }
        for rec in package.provenance_chain
    ]

    # Effective badge threshold
    badge_min = "verified" if require_verified else None

    # Run governor validation
    result = validate_import(
        blueprint=package.blueprint,
        generated_code=package.generated_code,
        provenance_chain=provenance_dicts,
        trust_score=package.trust_score,
        verification_badge=package.verification_badge,
        fidelity_scores=package.fidelity_scores,
        fingerprint=package.fingerprint,
        package_id=package.package_id,
        min_trust_score=min_trust_score,
        file_extension=package.file_extension,
    )

    # Additional badge check if require_verified
    if result.allowed and require_verified and package.verification_badge != "verified":
        return ImportValidationResult(
            allowed=False,
            rejection_reason="Badge is not 'verified' (--require-verified flag)",
            provenance_valid=result.provenance_valid,
            trust_sufficient=False,
            code_safe=result.code_safe,
            warnings=result.warnings,
            provenance_depth=result.provenance_depth,
            trust_score=result.trust_score,
            checks_performed=result.checks_performed + ("badge_requirement",),
        )

    if not result.allowed:
        # Log rejection for learning
        try:
            from core.rejection_log import RejectionLog, RejectionEvent
            from datetime import datetime, timezone
            log = RejectionLog()
            event = RejectionEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                package_id=package.package_id,
                source_instance=(
                    package.provenance_chain[-1].instance_id
                    if package.provenance_chain else "unknown"
                ),
                rejection_reason=result.rejection_reason or "",
                failed_check=_identify_failed_check(result),
                trust_score=result.trust_score,
                provenance_depth=result.provenance_depth,
            )
            log.record(event)
        except Exception as e:
            logger.debug(f"Rejection logging skipped: {e}")

    if result.allowed:
        # Register in local registry
        registry.register_tool(package, is_local=False)
        registry.record_usage(package.package_id, "import", instance_id)

    return result


def _identify_failed_check(result) -> str:
    """Map ImportValidationResult fields to the failed check name."""
    if not result.provenance_valid:
        return "provenance"
    if not result.trust_sufficient:
        return "trust"
    if not result.code_safe:
        return "code_safety"
    reason = (result.rejection_reason or "").lower()
    if "fingerprint" in reason:
        return "fingerprint"
    if "package_id" in reason or "package id" in reason:
        return "package_id"
    if "blueprint" in reason:
        return "blueprint"
    return "unknown"
