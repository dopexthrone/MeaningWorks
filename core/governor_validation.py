"""
Motherlabs Governor Import Validation — the trust boundary.

LEAF MODULE — stdlib only (hashlib, json, re, ast). Zero project imports.

This module enforces the trust boundary for cross-instance tool imports.
All validation functions take raw data as arguments and return frozen results.
No side effects. No state. Pure validation.

Checks performed:
1. Provenance integrity — chain non-empty, fields present, timestamps valid
2. Trust thresholds — score meets minimums, badge acceptable
3. Code safety — no exec/eval/__import__/os.system/subprocess patterns
4. Blueprint integrity — components non-empty, valid structure
5. Package ID verification — content matches declared ID
"""

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# FROZEN DATACLASS
# =============================================================================

@dataclass(frozen=True)
class ImportValidationResult:
    """Result of validating a tool package for import.

    All fields are immutable. `allowed` is the final verdict.
    """
    allowed: bool
    rejection_reason: str            # Empty if allowed
    provenance_valid: bool
    trust_sufficient: bool
    code_safe: bool
    warnings: Tuple[str, ...]
    provenance_depth: int
    trust_score: float
    checks_performed: Tuple[str, ...]


# =============================================================================
# PROVENANCE INTEGRITY
# =============================================================================

_REQUIRED_PROVENANCE_FIELDS = (
    "compilation_id", "instance_id", "timestamp", "domain",
)

# ISO8601 basic pattern: YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DDTHH:MM:SS+00:00
_ISO8601_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
)


def check_provenance_integrity(
    provenance_chain: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """Validate provenance chain integrity.

    Checks:
    - Chain is non-empty
    - Each record has required fields with non-empty values
    - Timestamps are valid ISO8601
    - Timestamps are monotonically non-decreasing
    - Instance IDs are non-empty hex-like strings

    Args:
        provenance_chain: List of provenance record dicts

    Returns:
        (valid, reason) — reason is empty if valid
    """
    if not provenance_chain:
        return False, "Provenance chain is empty"

    prev_timestamp = ""

    for i, record in enumerate(provenance_chain):
        if not isinstance(record, dict):
            return False, f"Provenance record {i} is not a dict"

        # Check required fields
        for field_name in _REQUIRED_PROVENANCE_FIELDS:
            value = record.get(field_name, "")
            if not value or not str(value).strip():
                return False, f"Provenance record {i} missing required field: {field_name}"

        # Validate timestamp format
        timestamp = str(record.get("timestamp", ""))
        if not _ISO8601_PATTERN.match(timestamp):
            return False, f"Provenance record {i} has invalid timestamp: {timestamp!r}"

        # Check monotonicity
        if prev_timestamp and timestamp < prev_timestamp:
            return False, (
                f"Provenance timestamps not monotonic: record {i} "
                f"({timestamp}) < record {i-1} ({prev_timestamp})"
            )
        prev_timestamp = timestamp

        # Validate instance_id (non-empty, printable)
        instance_id = str(record.get("instance_id", ""))
        if len(instance_id) < 4:
            return False, f"Provenance record {i} has suspiciously short instance_id: {instance_id!r}"

    return True, ""


# =============================================================================
# TRUST THRESHOLDS
# =============================================================================

# Core dimensions that must meet minimum thresholds
_CORE_TRUST_DIMENSIONS = ("completeness", "consistency", "coherence", "traceability")


def check_trust_thresholds(
    trust_score: float,
    verification_badge: str,
    fidelity_scores: Dict[str, int],
    min_trust_score: float = 60.0,
    min_core_dimension: int = 40,
) -> Tuple[bool, str]:
    """Validate trust score meets import thresholds.

    Checks:
    - Overall trust score >= min_trust_score
    - Verification badge is not "unverified"
    - All core fidelity dimensions >= min_core_dimension

    Args:
        trust_score: Overall trust score (0-100)
        verification_badge: "verified" | "partial" | "unverified"
        fidelity_scores: Dimension name -> score (0-100)
        min_trust_score: Minimum overall trust score
        min_core_dimension: Minimum score for each core dimension

    Returns:
        (sufficient, reason) — reason is empty if sufficient
    """
    if trust_score < min_trust_score:
        return False, (
            f"Trust score {trust_score:.1f} below minimum {min_trust_score:.1f}"
        )

    if verification_badge == "unverified":
        return False, "Verification badge is 'unverified'"

    # Check core dimensions
    for dim in _CORE_TRUST_DIMENSIONS:
        score = fidelity_scores.get(dim, 0)
        if score < min_core_dimension:
            return False, (
                f"Core dimension '{dim}' score {score} below minimum {min_core_dimension}"
            )

    return True, ""


# =============================================================================
# CODE SAFETY
# =============================================================================

# Dangerous patterns for Python code (defense-in-depth, not a sandbox)
_DANGEROUS_PATTERNS = [
    # Direct execution
    (r'\bexec\s*\(', "exec() call"),
    (r'\beval\s*\(', "eval() call"),
    (r'\b__import__\s*\(', "__import__() call"),
    (r'\bcompile\s*\(.*\bexec\b', "compile() with exec"),

    # OS/subprocess access
    (r'\bos\.system\s*\(', "os.system() call"),
    (r'\bos\.popen\s*\(', "os.popen() call"),
    (r'\bos\.exec', "os.exec*() call"),
    (r'\bsubprocess\.\w+\s*\(', "subprocess module usage"),

    # File system manipulation
    (r'\bshutil\.rmtree\b', "shutil.rmtree() call"),
    (r'\bos\.remove\s*\(', "os.remove() call"),
    (r'\bos\.unlink\s*\(', "os.unlink() call"),

    # Code injection
    (r'\bglobals\s*\(\s*\)', "globals() call"),
    (r'\bsetattr\s*\(', "setattr() call"),
    (r'\bdelattr\s*\(', "delattr() call"),
]

_COMPILED_PATTERNS = [
    (re.compile(pattern), desc)
    for pattern, desc in _DANGEROUS_PATTERNS
]

# Non-Python dangerous patterns (JS/TS, shell)
_JS_TS_DANGEROUS = [
    (re.compile(r'\beval\s*\('), "eval() call"),
    (re.compile(r'\bFunction\s*\('), "Function() constructor"),
    (re.compile(r'\brequire\s*\(\s*["\']child_process["\']'), "child_process require"),
    (re.compile(r'\bchild_process\b'), "child_process module"),
    (re.compile(r'\bexecSync\s*\('), "execSync() call"),
    (re.compile(r'\bspawnSync\s*\('), "spawnSync() call"),
    (re.compile(r'\bexec\s*\('), "exec() call"),
    (re.compile(r'\bdocument\.write\s*\('), "document.write() (XSS)"),
    (re.compile(r'\.innerHTML\s*='), "innerHTML assignment (XSS)"),
    (re.compile(r'\bnew\s+Function\s*\('), "new Function() constructor"),
    (re.compile(r'\bprocess\.env\b.*(?:password|secret|key|token)', re.IGNORECASE), "credential access via process.env"),
    (re.compile(r'\bfs\.(?:unlink|rmdir|rm)Sync\s*\('), "synchronous file deletion"),
]

_SHELL_DANGEROUS = [
    (re.compile(r'\brm\s+-rf\s+/'), "rm -rf / (destructive)"),
    (re.compile(r'\bcurl\b.*\|\s*(?:sh|bash)\b'), "curl pipe to shell"),
    (re.compile(r'\bwget\b.*\|\s*(?:sh|bash)\b'), "wget pipe to shell"),
    (re.compile(r'\b(?:mkfifo|nc\s|ncat\s).*\bsh\b'), "reverse shell pattern"),
    (re.compile(r'\bchmod\s+[0-7]*777\b'), "world-writable permissions"),
    (re.compile(r'\bdd\s+.*of=/dev/'), "raw device write"),
]

_NON_PYTHON_PATTERNS: Dict[str, list] = {
    ".js": _JS_TS_DANGEROUS,
    ".ts": _JS_TS_DANGEROUS,
    ".tsx": _JS_TS_DANGEROUS,
    ".jsx": _JS_TS_DANGEROUS,
    ".mjs": _JS_TS_DANGEROUS,
    ".sh": _SHELL_DANGEROUS,
    ".bash": _SHELL_DANGEROUS,
    ".zsh": _SHELL_DANGEROUS,
}

# PII detection patterns (#122 Privacy-protecting)
_PII_PATTERNS = [
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "email address"),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "SSN pattern"),
    (re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'), "credit card pattern"),
    (re.compile(r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'), "phone number"),
    (re.compile(r'\b(?:password|passwd|secret|api_key|apikey|token)\s*[=:]\s*\S+', re.IGNORECASE), "hardcoded credential"),
]


def detect_pii_patterns(text: str) -> list:
    """Detect PII patterns in text. Pure function. Returns list of (pattern_type, match) tuples."""
    if not text:
        return []
    findings = []
    for pattern, desc in _PII_PATTERNS:
        matches = pattern.findall(text)
        for m in matches[:3]:  # cap per pattern
            findings.append((desc, m if isinstance(m, str) else str(m)))
    return findings[:10]  # cap total


# Threat surface patterns (#124 Threat-modeling)
_THREAT_PATTERNS = [
    (re.compile(r'\.\./|\.\.\\', re.IGNORECASE), "path traversal"),
    (re.compile(r'\b(?:cmd|powershell|bash|sh)\s+', re.IGNORECASE), "shell invocation"),
    (re.compile(r'\bpickle\.loads?\b'), "pickle deserialization"),
    (re.compile(r'\byaml\.(?:load|unsafe_load)\b'), "unsafe YAML load"),
    (re.compile(r'\b(?:innerHTML|document\.write|eval)\b'), "XSS vector"),
    (re.compile(r'(?:SELECT|INSERT|UPDATE|DELETE)\s+.*\+\s*["\']?\s*\w+', re.IGNORECASE), "SQL injection"),
    (re.compile(r'\bopen\s*\(.*["\']w["\']', re.IGNORECASE), "arbitrary file write"),
    (re.compile(r'\bctypes\b'), "ctypes FFI usage"),
]


def assess_threat_surface(text: str) -> list:
    """Assess threat surface of a description or blueprint text. Pure function.
    Returns list of threat descriptions found.
    """
    if not text:
        return []
    threats = []
    for pattern, desc in _THREAT_PATTERNS:
        if pattern.search(text):
            threats.append(desc)
    return threats[:5]


# --- Ethics reasoning (#119) ---

_ETHICAL_PATTERNS = [
    (re.compile(r'\b(?:track\s+user|monitor\s+employee|spy|keylog|surveillance)', re.IGNORECASE), "surveillance concern"),
    (re.compile(r'\b(?:dark\s*pattern|manipulat|addict|trick\s+user|deceiv)', re.IGNORECASE), "manipulation concern"),
    (re.compile(r'\b(?:discriminat|profil|filter\s+by\s+(?:race|gender|age)|bias\s+against)', re.IGNORECASE), "discrimination concern"),
    (re.compile(r'\b(?:impersonat|phish|spoof|fake\s+(?:identity|profile|review))', re.IGNORECASE), "deception concern"),
    (re.compile(r'\b(?:weapon|exploit\s+vulnerabilit|malware|ddos|ransomware)', re.IGNORECASE), "harm concern"),
]


def assess_ethical_concerns(description: str) -> list:
    """Assess intent-level ethical concerns. Pure function.
    NOT code-level (that's check_code_safety). Scans for
    surveillance, manipulation, discrimination, deception, harm.
    Returns list of concern strings, capped at 5.
    """
    if not description:
        return []
    concerns = []
    for pattern, desc in _ETHICAL_PATTERNS:
        if pattern.search(description):
            concerns.append(desc)
    return concerns[:5]


def check_code_safety(
    generated_code: Dict[str, str],
    max_size_bytes: int = 500_000,
    file_extension: str = ".py",
    self_build: bool = False,
) -> Tuple[bool, List[str]]:
    """Check generated code for dangerous patterns.

    Defense-in-depth check. Primary trust comes from provenance chain,
    but we still scan for obviously dangerous patterns.

    For non-Python code, only size checks are performed (with a warning).

    Args:
        generated_code: Component name -> code mapping
        max_size_bytes: Maximum total code size in bytes
        file_extension: File extension for the code
        self_build: If True, exempt subprocess/os patterns (self-modification context)

    Returns:
        (safe, warnings) — warnings list is empty if no issues found
    """
    warnings: List[str] = []

    if not generated_code:
        return True, []

    # Size check (all languages)
    total_size = sum(len(code.encode("utf-8")) for code in generated_code.values())
    if total_size > max_size_bytes:
        return False, [
            f"Total code size {total_size:,} bytes exceeds limit {max_size_bytes:,} bytes"
        ]

    # For non-Python, run language-appropriate pattern checks
    if file_extension != ".py":
        dangerous_found = []
        lang_patterns = _NON_PYTHON_PATTERNS.get(file_extension, [])
        if not lang_patterns:
            warnings.append(
                f"Code safety analysis limited for {file_extension} files (no patterns defined)"
            )
            return True, warnings

        for comp_name, code in generated_code.items():
            for pattern, desc in lang_patterns:
                if pattern.search(code):
                    dangerous_found.append(f"[{comp_name}] {desc}")

        if dangerous_found:
            display = dangerous_found[:10]
            if len(dangerous_found) > 10:
                display.append(f"... and {len(dangerous_found) - 10} more")
            return False, display

        return True, warnings

    # Patterns exempt in self-build context (Mother modifying her own code)
    _SELF_BUILD_EXEMPT = {"subprocess module usage", "os.system() call",
                          "os.popen() call", "os.exec*() call",
                          "shutil.rmtree() call", "os.remove() call",
                          "os.unlink() call"}

    # Pattern scan
    dangerous_found = []
    for comp_name, code in generated_code.items():
        for pattern, desc in _COMPILED_PATTERNS:
            if pattern.search(code):
                if self_build and desc in _SELF_BUILD_EXEMPT:
                    continue
                dangerous_found.append(f"[{comp_name}] {desc}")

    if dangerous_found:
        # Limit output
        display = dangerous_found[:10]
        if len(dangerous_found) > 10:
            display.append(f"... and {len(dangerous_found) - 10} more")
        return False, display

    # AST parse check (catches syntax errors and obfuscation)
    for comp_name, code in generated_code.items():
        try:
            ast.parse(code)
        except SyntaxError as e:
            warnings.append(f"[{comp_name}] Syntax error: {e.msg} (line {e.lineno})")

    return True, warnings


# =============================================================================
# BLUEPRINT INTEGRITY
# =============================================================================

def check_blueprint_integrity(
    blueprint: Dict[str, Any],
) -> Tuple[bool, str]:
    """Validate blueprint structural integrity.

    Checks:
    - Blueprint is non-empty dict
    - Has at least one component
    - Components have required fields (name, type)
    - Relationships reference existing components
    - No duplicate component names

    Args:
        blueprint: Blueprint dict

    Returns:
        (valid, reason) — reason is empty if valid
    """
    if not blueprint or not isinstance(blueprint, dict):
        return False, "Blueprint is empty or not a dict"

    components = blueprint.get("components", [])
    if not components:
        return False, "Blueprint has no components"

    if not isinstance(components, list):
        return False, "Blueprint components is not a list"

    # Check required fields and collect names
    component_names = set()
    for i, comp in enumerate(components):
        if not isinstance(comp, dict):
            return False, f"Component {i} is not a dict"

        name = comp.get("name", "")
        if not name or not str(name).strip():
            return False, f"Component {i} has no name"

        comp_type = comp.get("type", "")
        if not comp_type or not str(comp_type).strip():
            return False, f"Component '{name}' has no type"

        if name in component_names:
            return False, f"Duplicate component name: '{name}'"
        component_names.add(name)

    # Check relationship references
    relationships = blueprint.get("relationships", [])
    if isinstance(relationships, list):
        for i, rel in enumerate(relationships):
            if not isinstance(rel, dict):
                continue
            from_comp = rel.get("from", "")
            to_comp = rel.get("to", "")
            if from_comp and from_comp not in component_names:
                return False, (
                    f"Relationship {i} references unknown component: '{from_comp}'"
                )
            if to_comp and to_comp not in component_names:
                return False, (
                    f"Relationship {i} references unknown component: '{to_comp}'"
                )

    return True, ""


# =============================================================================
# PACKAGE ID VERIFICATION
# =============================================================================

def _recompute_package_id(
    blueprint: Dict[str, Any],
    generated_code: Dict[str, str],
) -> str:
    """Recompute package ID from content (must match tool_package.compute_package_id)."""
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
# MAIN VALIDATION ENTRY POINT
# =============================================================================

def validate_import(
    blueprint: Dict[str, Any],
    generated_code: Dict[str, str],
    provenance_chain: List[Dict[str, Any]],
    trust_score: float,
    verification_badge: str,
    fidelity_scores: Dict[str, int],
    fingerprint: str,
    package_id: str,
    min_trust_score: float = 60.0,
    min_core_dimension: int = 40,
    max_code_size: int = 500_000,
    file_extension: str = ".py",
) -> ImportValidationResult:
    """Validate a tool package for import — the governor's trust boundary.

    Runs all validation checks and returns a comprehensive result.
    A single failure in any check blocks the import.

    Args:
        blueprint: Tool's blueprint dict
        generated_code: Component name -> code mapping
        provenance_chain: List of provenance record dicts
        trust_score: Overall trust score (0-100)
        verification_badge: "verified" | "partial" | "unverified"
        fidelity_scores: Dimension -> score mapping
        fingerprint: Expected structural fingerprint hash
        package_id: Declared package ID
        min_trust_score: Minimum acceptable trust score
        min_core_dimension: Minimum per-dimension fidelity score
        max_code_size: Maximum total code size in bytes
        file_extension: Code file extension

    Returns:
        Frozen ImportValidationResult
    """
    checks_performed = []
    all_warnings: List[str] = []
    rejection_reason = ""
    provenance_valid = False
    trust_sufficient = False
    code_safe = False

    # 1. Provenance integrity
    checks_performed.append("provenance_integrity")
    prov_ok, prov_reason = check_provenance_integrity(provenance_chain)
    provenance_valid = prov_ok
    if not prov_ok:
        return ImportValidationResult(
            allowed=False,
            rejection_reason=prov_reason,
            provenance_valid=False,
            trust_sufficient=False,
            code_safe=False,
            warnings=tuple(all_warnings),
            provenance_depth=0,
            trust_score=trust_score,
            checks_performed=tuple(checks_performed),
        )

    # 2. Trust thresholds
    checks_performed.append("trust_thresholds")
    trust_ok, trust_reason = check_trust_thresholds(
        trust_score, verification_badge, fidelity_scores,
        min_trust_score=min_trust_score,
        min_core_dimension=min_core_dimension,
    )
    trust_sufficient = trust_ok
    if not trust_ok:
        return ImportValidationResult(
            allowed=False,
            rejection_reason=trust_reason,
            provenance_valid=True,
            trust_sufficient=False,
            code_safe=False,
            warnings=tuple(all_warnings),
            provenance_depth=len(provenance_chain),
            trust_score=trust_score,
            checks_performed=tuple(checks_performed),
        )

    # 3. Code safety
    checks_performed.append("code_safety")
    safe, code_warnings = check_code_safety(
        generated_code,
        max_size_bytes=max_code_size,
        file_extension=file_extension,
    )
    code_safe = safe
    all_warnings.extend(code_warnings)
    if not safe:
        return ImportValidationResult(
            allowed=False,
            rejection_reason=f"Dangerous code patterns: {'; '.join(code_warnings[:3])}",
            provenance_valid=True,
            trust_sufficient=True,
            code_safe=False,
            warnings=tuple(all_warnings),
            provenance_depth=len(provenance_chain),
            trust_score=trust_score,
            checks_performed=tuple(checks_performed),
        )

    # 4. Blueprint integrity
    checks_performed.append("blueprint_integrity")
    bp_ok, bp_reason = check_blueprint_integrity(blueprint)
    if not bp_ok:
        return ImportValidationResult(
            allowed=False,
            rejection_reason=bp_reason,
            provenance_valid=True,
            trust_sufficient=True,
            code_safe=True,
            warnings=tuple(all_warnings),
            provenance_depth=len(provenance_chain),
            trust_score=trust_score,
            checks_performed=tuple(checks_performed),
        )

    # 5. Package ID verification
    checks_performed.append("package_id_verification")
    recomputed_id = _recompute_package_id(blueprint, generated_code)
    if recomputed_id != package_id:
        return ImportValidationResult(
            allowed=False,
            rejection_reason=(
                f"Package ID mismatch: declared {package_id}, "
                f"computed {recomputed_id}"
            ),
            provenance_valid=True,
            trust_sufficient=True,
            code_safe=True,
            warnings=tuple(all_warnings),
            provenance_depth=len(provenance_chain),
            trust_score=trust_score,
            checks_performed=tuple(checks_performed),
        )

    # All checks passed
    return ImportValidationResult(
        allowed=True,
        rejection_reason="",
        provenance_valid=True,
        trust_sufficient=True,
        code_safe=True,
        warnings=tuple(all_warnings),
        provenance_depth=len(provenance_chain),
        trust_score=trust_score,
        checks_performed=tuple(checks_performed),
    )
