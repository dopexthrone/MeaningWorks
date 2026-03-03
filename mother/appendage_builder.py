"""
Appendage builder — scaffold generation, build prompt, and validation.

LEAF module. Stdlib only. No imports from core/ or mother/.

Generates the project directory and agent script scaffold for a new
appendage. Contains the build prompt template that instructs Claude Code
to implement the actual capability logic.
"""

import ast
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class BuildSpec:
    """Input specification for building an appendage."""

    name: str
    description: str
    capability_gap: str
    capabilities: List[str] = ()
    constraints: str = ""


_AGENT_SCAFFOLD = '''\
"""
{name} — appendage agent.

JSON Lines protocol handler. Communicates with Mother via stdin/stdout.
{description}
"""

import json
import sys


def handle_request(params: dict) -> dict:
    """Process a single request from Mother.

    Args:
        params: The request parameters from Mother.

    Returns:
        A dict with the result. Must be JSON-serializable.
    """
    return {{"output": "not implemented", "status": "stub"}}


def main():
    """Main loop: emit ready signal, then process requests."""
    # Announce capabilities
    ready = {{
        "id": "ready",
        "result": {{
            "capabilities": {capabilities},
            "version": "1.0",
        }},
    }}
    sys.stdout.write(json.dumps(ready) + "\\n")
    sys.stdout.flush()

    # Process requests
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        request_id = request.get("id", "unknown")
        params = request.get("params", {{}})

        try:
            result = handle_request(params)
            response = {{"id": request_id, "result": result, "error": None}}
        except Exception as e:
            response = {{"id": request_id, "result": None, "error": str(e)}}

        sys.stdout.write(json.dumps(response) + "\\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
'''


def scaffold_project_dir(base_dir: str, name: str) -> str:
    """Create the appendage project directory with scaffold main.py.

    Creates: <base_dir>/<name>/main.py

    Returns the project directory path.
    """
    project_dir = os.path.join(base_dir, name)
    os.makedirs(project_dir, exist_ok=True)

    main_path = os.path.join(project_dir, "main.py")
    if not os.path.exists(main_path):
        content = _AGENT_SCAFFOLD.format(
            name=name,
            description="",
            capabilities=[],
        )
        with open(main_path, "w") as f:
            f.write(content)

    return project_dir


def generate_build_prompt(spec: BuildSpec) -> str:
    """Generate the Claude Code prompt to implement the appendage.

    The prompt instructs Claude Code to:
    1. Read the existing scaffold main.py
    2. Implement handle_request() for the specific capability
    3. Update the capabilities list in the ready signal
    4. Keep the JSON Lines protocol intact
    """
    caps_str = ", ".join(f'"{c}"' for c in spec.capabilities)

    prompt = f"""Implement the '{spec.name}' agent in main.py.

WHAT IT DOES:
{spec.description}

CAPABILITY GAP IT FILLS:
{spec.capability_gap}

CAPABILITIES TO ADVERTISE:
[{caps_str}]

REQUIREMENTS:
1. Read main.py — it already has the JSON Lines protocol scaffold.
2. Implement the handle_request(params) function to perform the actual work.
3. Update the capabilities list in the ready signal to: [{caps_str}]
4. DO NOT modify the protocol structure (ready signal format, request/response JSON format).
5. Use only stdlib modules. No pip install.
6. The agent runs as a long-lived subprocess — it must handle multiple requests.
7. handle_request must return a JSON-serializable dict.
8. Handle errors gracefully — return informative error messages, don't crash."""

    if spec.constraints:
        prompt += f"\n\nADDITIONAL CONSTRAINTS:\n{spec.constraints}"

    prompt += """

PROTOCOL REMINDER:
- Startup: write {"id": "ready", "result": {"capabilities": [...]}} to stdout
- Request: read {"id": "req_001", "method": "invoke", "params": {...}} from stdin
- Response: write {"id": "req_001", "result": {...}, "error": null} to stdout
- All messages are single-line JSON followed by newline.
- Always flush stdout after writing."""

    return prompt


def validate_agent_script(
    project_dir: str,
    entry_point: str = "main.py",
) -> Tuple[bool, str]:
    """Validate an agent script for syntax and protocol compliance.

    Checks:
    1. File exists
    2. Valid Python syntax (compile)
    3. Protocol compliance: imports json, has ready signal pattern

    Returns (valid, message).
    """
    script_path = os.path.join(project_dir, entry_point)

    if not os.path.isfile(script_path):
        return False, f"File not found: {script_path}"

    with open(script_path, "r") as f:
        source = f.read()

    # Syntax check
    try:
        ast.parse(source, filename=entry_point)
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"

    # Protocol compliance: must import json
    if "import json" not in source:
        return False, "Missing 'import json' — required for JSON Lines protocol"

    # Protocol compliance: must have ready signal
    if '"ready"' not in source and "'ready'" not in source:
        return False, "Missing ready signal — agent must emit {\"id\": \"ready\", ...} on startup"

    # Protocol compliance: must read from stdin
    if "sys.stdin" not in source and "stdin" not in source:
        return False, "Missing stdin reader — agent must read requests from stdin"

    return True, "Valid"
