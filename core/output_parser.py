"""
Structured output parser with schema validation.

Phase 22: Quality Gate Tightening.

LEAF MODULE — stdlib only (json, re, dataclasses, typing).
No project imports. This module provides:
- JSON extraction (3 strategies: direct, code block, brace)
- Schema validation (required fields, type checks, min_length)
- Repair prompt construction for LLM retry

Used by engine._parse_structured_output() to replace bare _extract_json() calls.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ParseResult:
    """Result of structured output parsing."""

    success: bool
    data: Dict[str, Any]
    errors: Tuple[str, ...]
    raw_text: str
    repair_hint: str = ""


@dataclass(frozen=True)
class FieldSpec:
    """Specification for a single field in a stage schema."""

    name: str
    required: bool = True
    expected_type: str = "any"  # "str", "list", "dict", "number", "bool", "any"
    min_length: int = 0  # for str/list — minimum length to be valid


@dataclass(frozen=True)
class StageSchema:
    """Schema for a compilation stage's expected JSON output."""

    name: str
    fields: Tuple[FieldSpec, ...]
    allows_array: bool = False  # if True, bare JSON array is acceptable


# =========================================================================
# PRE-BUILT SCHEMAS
# =========================================================================

STAGE_SCHEMAS: Dict[str, StageSchema] = {
    "intent": StageSchema(
        name="intent",
        fields=(
            FieldSpec(name="core_need", required=True, expected_type="str", min_length=3),
            FieldSpec(name="actors", required=False, expected_type="list"),
            FieldSpec(name="constraints", required=False, expected_type="list"),
            FieldSpec(name="domain", required=False, expected_type="str"),
        ),
    ),
    "personas": StageSchema(
        name="personas",
        fields=(
            FieldSpec(name="personas", required=True, expected_type="list", min_length=1),
        ),
        allows_array=True,
    ),
    "synthesis": StageSchema(
        name="synthesis",
        fields=(
            FieldSpec(name="components", required=True, expected_type="list", min_length=1),
            FieldSpec(name="relationships", required=False, expected_type="list"),
            FieldSpec(name="constraints", required=False, expected_type="list"),
            FieldSpec(name="unresolved", required=False, expected_type="list"),
        ),
    ),
    "verification": StageSchema(
        name="verification",
        fields=(
            FieldSpec(name="status", required=True, expected_type="str"),
            FieldSpec(name="completeness", required=False, expected_type="number"),
            FieldSpec(name="consistency", required=False, expected_type="number"),
            FieldSpec(name="traceability", required=False, expected_type="number"),
        ),
    ),
}


# =========================================================================
# JSON EXTRACTION
# =========================================================================

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from text using 3 strategies.

    Same logic as engine._extract_json() but standalone.
    Handles bare arrays by wrapping in a dict if the schema allows it.

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    text = text.strip()

    # Strategy 1: Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return {"_array": result}
        return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract from markdown code block
    code_block_match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n```', text)
    if code_block_match:
        try:
            result = json.loads(code_block_match.group(1).strip())
            if isinstance(result, list):
                return {"_array": result}
            return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Find outermost braces
    start = text.find('{')
    end = text.rfind('}') + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3b: Find outermost brackets (array)
    start = text.find('[')
    end = text.rfind(']') + 1
    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, list):
                return {"_array": result}
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError("Could not extract JSON from response")


# =========================================================================
# SCHEMA VALIDATION
# =========================================================================

_TYPE_CHECKS = {
    "str": lambda v: isinstance(v, str),
    "list": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "any": lambda v: True,
}


def validate_against_schema(data: Dict[str, Any], schema: StageSchema) -> Tuple[str, ...]:
    """
    Validate parsed data against a stage schema.

    Returns:
        Tuple of error strings (empty if valid).
    """
    errors = []

    for field in schema.fields:
        value = data.get(field.name)

        if value is None:
            if field.required:
                errors.append(f"Missing required field: '{field.name}'")
            continue

        # Type check
        checker = _TYPE_CHECKS.get(field.expected_type, _TYPE_CHECKS["any"])
        if not checker(value):
            errors.append(
                f"Field '{field.name}' expected type '{field.expected_type}', "
                f"got '{type(value).__name__}'"
            )
            continue

        # Min length check (for str and list)
        if field.min_length > 0:
            if isinstance(value, (str, list)) and len(value) < field.min_length:
                errors.append(
                    f"Field '{field.name}' too short: length {len(value)}, "
                    f"minimum {field.min_length}"
                )

    return tuple(errors)


def _build_repair_hint(errors: Tuple[str, ...]) -> str:
    """Build a human-readable repair hint from validation errors."""
    parts = []
    for err in errors:
        if "Missing required field" in err:
            field_name = err.split("'")[1]
            parts.append(f"Include the '{field_name}' field in your JSON output.")
        elif "expected type" in err:
            parts.append(f"Fix type error: {err}")
        elif "too short" in err:
            parts.append(f"Provide more content: {err}")
        else:
            parts.append(err)
    return " ".join(parts)


# =========================================================================
# ORCHESTRATOR
# =========================================================================

def parse_structured_output(text: str, schema: StageSchema) -> ParseResult:
    """
    Parse and validate structured LLM output against a schema.

    1. Extract JSON (3 strategies)
    2. Handle array wrapping for schemas that allow it
    3. Validate against schema
    4. Return ParseResult with success/errors/repair_hint
    """
    # Step 1: Extract JSON
    try:
        data = extract_json(text)
    except ValueError:
        return ParseResult(
            success=False,
            data={},
            errors=("JSON extraction failed: no valid JSON found in response.",),
            raw_text=text,
            repair_hint="Your response must contain valid JSON. "
                        "Output a JSON object with the required fields.",
        )

    # Step 2: Handle array wrapping
    if "_array" in data and schema.allows_array:
        # Find the list-type required field to assign the array to
        list_fields = [f for f in schema.fields if f.expected_type == "list" and f.required]
        if list_fields:
            data = {list_fields[0].name: data["_array"]}
        else:
            data = {"items": data["_array"]}
    elif "_array" in data:
        # Schema doesn't allow arrays — this is an error
        return ParseResult(
            success=False,
            data={},
            errors=("Expected a JSON object, got a JSON array.",),
            raw_text=text,
            repair_hint="Output a JSON object (with curly braces), not an array.",
        )

    # Step 3: Validate
    errors = validate_against_schema(data, schema)

    if errors:
        return ParseResult(
            success=False,
            data=data,
            errors=errors,
            raw_text=text,
            repair_hint=_build_repair_hint(errors),
        )

    return ParseResult(
        success=True,
        data=data,
        errors=(),
        raw_text=text,
    )


# =========================================================================
# REPAIR PROMPT
# =========================================================================

def build_repair_prompt(result: ParseResult, context: str = "") -> str:
    """
    Build a retry prompt for LLM when parsing/validation failed.

    Args:
        result: Failed ParseResult with errors and repair_hint
        context: Optional context (e.g., truncated original prompt)

    Returns:
        Prompt string to send back to LLM for retry.
    """
    error_list = "\n".join(f"- {e}" for e in result.errors)
    parts = [
        "Your previous response could not be parsed correctly.",
        f"Errors:\n{error_list}",
    ]
    if result.repair_hint:
        parts.append(f"Hint: {result.repair_hint}")
    parts.append("Please output valid JSON with the required structure.")
    if context:
        parts.append(f"Original request context: {context[:500]}")
    return "\n\n".join(parts)
