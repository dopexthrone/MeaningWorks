"""
Motherlabs Project Writer — write runnable projects to disk.

Phase 1 of Agent Ship: Takes Dict[str, str] generated code + blueprint
and writes a complete runnable Python project structure.

This is a LEAF MODULE — stdlib only. No engine/protocol/pipeline imports.
"""

import ast
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from core.naming import sanitize_name, to_snake, to_pascal, slugify

# Re-exports for backward compatibility (tests import these from here)
_sanitize_name = sanitize_name
_to_pascal = to_pascal
_to_snake = to_snake
_slugify = slugify


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class ProjectConfig:
    """Configuration for project writing."""
    language: str = "python"
    project_name: str = ""          # Inferred from blueprint if empty
    entry_point: bool = True
    tests: bool = True
    per_component_files: int = 6    # Threshold: >N components → one file each
    runtime_capabilities: Optional[Any] = None  # RuntimeCapabilities from domain adapter
    clean_before_write: bool = True  # Remove stale files from previous builds


@dataclass(frozen=True)
class ProjectManifest:
    """Result of writing a project to disk."""
    project_dir: str
    files_written: tuple             # Relative paths from project_dir
    entry_point: str                 # Relative path to main.py
    total_lines: int
    file_contents: Dict[str, str] = field(default_factory=dict)  # filename → content for API
    cross_module_warnings: Tuple[str, ...] = ()  # advisory only


# =============================================================================
# PROJECT NAME INFERENCE
# =============================================================================

def _infer_project_name(blueprint: Dict[str, Any]) -> str:
    """Infer project name from blueprint domain/core_need.

    Tries: domain → core_need first line → "project".
    Slugifies to valid Python package name.
    """
    # Try domain field
    domain = blueprint.get("domain", "")
    if domain and domain.strip():
        return _slugify(domain)

    # Try core_need first meaningful phrase
    core_need = blueprint.get("core_need", "")
    if core_need:
        # Take first line or first 4 words
        first_line = core_need.strip().split("\n")[0]
        words = first_line.split()[:4]
        if words:
            return _slugify(" ".join(words))

    return "project"


# =============================================================================
# COMPONENT GROUPING
# =============================================================================

_ENTITY_TYPES = frozenset({"entity", "data", "model", "record", "state", "store"})
_PROCESS_TYPES = frozenset({"process", "agent", "service", "handler", "controller",
                            "manager", "orchestrator", "pipeline", "workflow"})


def _get_component_type(name: str, blueprint: Dict[str, Any]) -> str:
    """Look up component type from blueprint."""
    for comp in blueprint.get("components", []):
        if comp.get("name") == name:
            return comp.get("type", "entity").lower()
    return "entity"


def _group_components(
    generated_code: Dict[str, str],
    blueprint: Dict[str, Any],
    threshold: int = 6,
    entity_types: Optional[frozenset] = None,
    file_extension: str = ".py",
) -> Dict[str, Dict[str, str]]:
    """Group components into files by type.

    If total components <= threshold: models{ext} + services{ext}
    If total components > threshold: one file per component

    Args:
        generated_code: Component name → code mapping
        blueprint: Compiled blueprint dict
        threshold: Component count threshold for per-file grouping
        entity_types: Optional domain-specific entity types (default: software)
        file_extension: Output file extension (default: ".py")

    Returns: {filename: {component_name: code}}
    """
    ent_types = entity_types if entity_types is not None else _ENTITY_TYPES
    if len(generated_code) > threshold:
        # Per-component files
        result = {}
        for name, code in generated_code.items():
            filename = _slugify(name) + file_extension
            result[filename] = {name: code}
        return result

    # Group into models vs services
    models: Dict[str, str] = {}
    services: Dict[str, str] = {}

    for name, code in generated_code.items():
        comp_type = _get_component_type(name, blueprint)
        if comp_type in ent_types:
            models[name] = code
        else:
            services[name] = code

    result = {}
    if models:
        result["models" + file_extension] = models
    if services:
        result["services" + file_extension] = services

    # If all components ended up in one bucket, just use it
    if not result and generated_code:
        result["components" + file_extension] = dict(generated_code)

    return result


# =============================================================================
# IMPORT RESOLUTION
# =============================================================================

def _strip_llm_relative_imports(code: str) -> str:
    """Strip relative imports (from .foo import Bar) from LLM-emitted code.

    project_writer._resolve_imports() generates correct imports based on
    actual file layout. LLM-emitted relative imports from prompt hints
    conflict. Strip them so _resolve_imports is the single source.
    """
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("from .") and "import" in stripped:
            continue  # Drop LLM-generated relative imports
        cleaned.append(line)
    return "\n".join(cleaned)


def _sanitize_concatenated_code(code: str) -> str:
    """Sanitize code produced by concatenating multiple component blocks.

    Fixes:
    0. Strips LLM-generated relative imports (from .foo import Bar)
    1. Hoists `from __future__ import ...` to the top of the file
       (Python requires these before any other code)
    2. Deduplicates import lines that appear multiple times
    3. Removes duplicate blank-line runs

    Args:
        code: Concatenated Python source

    Returns:
        Sanitized source with valid import ordering
    """
    # Strip LLM-generated relative imports before deduplication
    code = _strip_llm_relative_imports(code)
    lines = code.split("\n")

    future_imports: List[str] = []
    regular_imports: List[str] = []
    body_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("from __future__ import"):
            if stripped not in future_imports:
                future_imports.append(stripped)
        elif stripped.startswith(("import ", "from ")) and not body_lines:
            # Import at top of a block — deduplicate
            if stripped not in regular_imports:
                regular_imports.append(line)
        else:
            body_lines.append(line)

    # Rebuild: __future__ first, then regular imports, then body
    # But body may have more import lines interspersed (from concatenation)
    # Do a second pass: pull all remaining __future__ from body
    clean_body: List[str] = []
    seen_imports: set = set(s.strip() for s in regular_imports)
    for line in body_lines:
        stripped = line.strip()
        if stripped.startswith("from __future__ import"):
            if stripped not in future_imports:
                future_imports.append(stripped)
            continue
        # Deduplicate top-level import lines from subsequent blocks
        if stripped.startswith(("import ", "from ")) and stripped in seen_imports:
            continue
        if stripped.startswith(("import ", "from ")):
            seen_imports.add(stripped)
        clean_body.append(line)

    parts: List[str] = []
    if future_imports:
        parts.extend(future_imports)
        parts.append("")
    if regular_imports:
        parts.extend(regular_imports)
        parts.append("")
    parts.extend(clean_body)

    return "\n".join(parts)


def _resolve_imports(
    grouped_files: Dict[str, Dict[str, str]],
    all_component_names: List[str],
) -> Dict[str, str]:
    """Resolve cross-file imports for generated code.

    Scans each file's code for references to component class names
    that live in other files. Adds appropriate from-imports.

    Returns: {filename: full file content with imports}
    """
    # Map component name → filename
    name_to_file: Dict[str, str] = {}
    for filename, components in grouped_files.items():
        for name in components:
            name_to_file[name] = filename

    # Build class name patterns (PascalCase versions)
    class_names = set()
    for name in all_component_names:
        class_names.add(name)
        # Also add PascalCase variant
        pascal = _to_pascal(name)
        class_names.add(pascal)

    result: Dict[str, str] = {}

    for filename, components in grouped_files.items():
        combined_code = _sanitize_concatenated_code(
            "\n\n".join(components.values())
        )

        # Find references to components in other files
        imports_needed: Dict[str, List[str]] = {}  # source_file → [class_names]

        for other_name in all_component_names:
            other_file = name_to_file.get(other_name, "")
            if other_file == filename or not other_file:
                continue

            # Check if this component's name appears in our code
            pascal = _to_pascal(other_name)
            if other_name in combined_code or pascal in combined_code:
                module = other_file.replace(".py", "")
                if module not in imports_needed:
                    imports_needed[module] = []
                imports_needed[module].append(pascal)

        # Build import header
        import_lines = []
        for module, names in sorted(imports_needed.items()):
            unique_names = sorted(set(names))
            import_lines.append(f"from .{module} import {', '.join(unique_names)}")

        if import_lines:
            result[filename] = "\n".join(import_lines) + "\n\n\n" + combined_code
        else:
            result[filename] = combined_code

    return result


# =============================================================================
# CIRCULAR IMPORT DETECTION & BREAKING
# =============================================================================

def _detect_import_cycles(resolved_files: Dict[str, str]) -> List[List[str]]:
    """Detect circular import chains in resolved files.

    Parses each file's AST, builds import graph from ImportFrom nodes
    with level==1 (relative imports), runs DFS with coloring to find cycles.

    Returns list of cycles, each cycle is a list of module names forming the loop.
    """
    # Build adjacency from relative imports
    graph: Dict[str, set] = {}
    for filename, content in resolved_files.items():
        module = filename.replace(".py", "")
        graph.setdefault(module, set())
        try:
            tree = ast.parse(content, filename=filename)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                target = node.module
                if target in graph or any(
                    f.replace(".py", "") == target for f in resolved_files
                ):
                    graph.setdefault(target, set())
                    graph[module].add(target)

    # DFS with WHITE/GRAY/BLACK coloring
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}
    cycles: List[List[str]] = []

    def dfs(node: str, path: List[str]) -> None:
        color[node] = GRAY
        path.append(node)
        for neighbor in graph.get(node, set()):
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY and neighbor in path:
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
            elif color[neighbor] == WHITE:
                dfs(neighbor, path)
        path.pop()
        color[node] = BLACK

    for node in list(graph.keys()):
        if color.get(node) == WHITE:
            dfs(node, [])

    return cycles


def _break_import_cycles(
    resolved_files: Dict[str, str],
    cycles: List[List[str]],
) -> Dict[str, str]:
    """Break circular imports using TYPE_CHECKING guards.

    For each cycle, identifies the back-edge (the import that closes the cycle).
    Picks the edge where the importing module references the imported name least.
    Converts that import to a TYPE_CHECKING-guarded import.
    """
    result = dict(resolved_files)

    for cycle in cycles:
        if len(cycle) < 3:  # Need at least [A, B, A]
            continue

        # The back-edge is the last edge: cycle[-2] imports from cycle[-1] (== cycle[0])
        # But we want to pick the edge with fewest runtime references
        best_edge = None
        best_count = float('inf')

        for i in range(len(cycle) - 1):
            importer = cycle[i]
            imported = cycle[i + 1]
            importer_file = importer + ".py"
            if importer_file not in result:
                continue

            # Find what names are imported from the target module
            content = result[importer_file]
            try:
                tree = ast.parse(content, filename=importer_file)
            except SyntaxError:
                continue

            imported_names: List[str] = []
            for node in ast.walk(tree):
                if (isinstance(node, ast.ImportFrom) and node.level == 1
                        and node.module == imported):
                    for alias in node.names:
                        imported_names.append(alias.name)

            if not imported_names:
                continue

            # Count runtime references (excluding import lines and type annotations)
            ref_count = 0
            for name in imported_names:
                # Count occurrences in the code body (rough but sufficient)
                ref_count += content.count(name) - 1  # subtract the import itself

            if ref_count < best_count:
                best_count = ref_count
                best_edge = (importer, imported, imported_names)

        if best_edge is None:
            continue

        importer, imported, names = best_edge
        importer_file = importer + ".py"
        content = result[importer_file]

        # Build the import line pattern to replace
        lines = content.split("\n")
        new_lines: List[str] = []
        guard_names: List[str] = []
        has_future_annotations = any(
            "from __future__ import annotations" in line for line in lines
        )
        has_type_checking = any(
            "from typing import TYPE_CHECKING" in line
            or "TYPE_CHECKING" in line
            for line in lines
        )

        for line in lines:
            stripped = line.strip()
            # Match the specific relative import we want to guard
            if (stripped.startswith(f"from {imported} import")
                    or stripped.startswith(f"from .{imported} import")):
                # Extract names from this import line
                match = re.match(
                    r'from \.?' + re.escape(imported) + r'\s+import\s+(.+)',
                    stripped,
                )
                if match:
                    guard_names.extend(
                        n.strip() for n in match.group(1).split(",")
                    )
                    continue  # Remove this line — will add in TYPE_CHECKING block
            new_lines.append(line)

        if not guard_names:
            continue

        # Build new content with TYPE_CHECKING guard
        header: List[str] = []
        if not has_future_annotations:
            header.append("from __future__ import annotations")
        if not has_type_checking:
            header.append("from typing import TYPE_CHECKING")

        guard_block = [
            "",
            "if TYPE_CHECKING:",
            f"    from {imported} import {', '.join(sorted(set(guard_names)))}",
            "",
        ]

        # Insert header at top (after any existing future imports)
        final_lines: List[str] = []
        header_inserted = False
        guard_inserted = False

        for line in new_lines:
            if not header_inserted and header:
                stripped = line.strip()
                # Insert before first non-empty, non-future-import, non-comment line
                if (stripped and not stripped.startswith("#")
                        and not stripped.startswith("from __future__")
                        and not stripped.startswith('"""')
                        and not stripped.startswith("'''")):
                    for h in header:
                        final_lines.append(h)
                    final_lines.append("")
                    header_inserted = True

            final_lines.append(line)

            # Insert guard block after all regular imports
            if (not guard_inserted and line.strip().startswith(("import ", "from "))
                    and not line.strip().startswith("from __future__")
                    and not line.strip().startswith("from typing import TYPE_CHECKING")):
                # Peek ahead — if next non-empty line is not an import, insert guard
                pass

        # Simpler: just append guard block after last import line
        if not guard_inserted:
            insert_pos = 0
            for i, line in enumerate(final_lines):
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and not stripped.startswith("from __future__"):
                    insert_pos = i + 1
            for j, gl in enumerate(guard_block):
                final_lines.insert(insert_pos + j, gl)

        if not header_inserted and header:
            # No good insertion point found — prepend
            final_lines = header + [""] + final_lines

        result[importer_file] = "\n".join(final_lines)

    return result


# =============================================================================
# INTERFACE RECONCILIATION
# =============================================================================

def _reconcile_interfaces(
    resolved_files: Dict[str, str],
) -> Tuple[Dict[str, str], List[str]]:
    """Fix method call mismatches between emitted components.

    For each method call on a known imported type, checks if that method
    exists in the target class. If not, finds the closest match and patches.

    Returns (patched_files, list_of_fixes_applied).
    """
    # 1. Build class_methods: {class_name: set(method_names)}
    class_methods: Dict[str, set] = {}
    for filename, content in resolved_files.items():
        try:
            tree = ast.parse(content, filename=filename)
        except SyntaxError:
            continue
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                methods = set()
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.add(item.name)
                class_methods[node.name] = methods

    # 2. Build variable → class type mapping from constructor calls and type hints
    fixes: List[str] = []
    result = dict(resolved_files)

    for filename, content in resolved_files.items():
        try:
            tree = ast.parse(content, filename=filename)
        except SyntaxError:
            continue

        # Map variable names to class types within each class/function scope
        var_types: Dict[str, str] = {}

        for node in ast.walk(tree):
            # Track assignments like: self.foo = Foo() or foo = Foo()
            if isinstance(node, ast.Assign):
                if (isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id in class_methods):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute):
                            var_types[target.attr] = node.value.func.id
                        elif isinstance(target, ast.Name):
                            var_types[target.id] = node.value.func.id
            # Track __init__ params with type annotations
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    if arg.annotation and isinstance(arg.annotation, ast.Name):
                        if arg.annotation.id in class_methods:
                            var_types[arg.arg] = arg.annotation.id
                    # Also handle string annotations
                    if arg.annotation and isinstance(arg.annotation, ast.Constant):
                        ann = str(arg.annotation.value)
                        if ann in class_methods:
                            var_types[arg.arg] = ann

        # 3. Find method calls on typed variables that don't exist
        patched_content = content
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
                # self.foo.method() pattern
                obj_name = node.value.attr
                method_name = node.attr
                if obj_name in var_types:
                    cls = var_types[obj_name]
                    if cls in class_methods and method_name not in class_methods[cls]:
                        match = _find_closest_method(method_name, class_methods[cls])
                        if match:
                            old = f".{method_name}("
                            new = f".{match}("
                            if old in patched_content:
                                patched_content = patched_content.replace(old, new)
                                fixes.append(
                                    f"{filename}: {cls}.{method_name}() → {cls}.{match}()"
                                )
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                # foo.method() pattern
                obj_name = node.value.id
                method_name = node.attr
                if obj_name in var_types:
                    cls = var_types[obj_name]
                    if cls in class_methods and method_name not in class_methods[cls]:
                        match = _find_closest_method(method_name, class_methods[cls])
                        if match:
                            old = f".{method_name}("
                            new = f".{match}("
                            if old in patched_content:
                                patched_content = patched_content.replace(old, new)
                                fixes.append(
                                    f"{filename}: {cls}.{method_name}() → {cls}.{match}()"
                                )

        if patched_content != content:
            # Strip # type: ignore[attr-defined] from fixed lines
            patched_content = re.sub(
                r'\s*#\s*type:\s*ignore\[attr-defined\]', '', patched_content,
            )
            result[filename] = patched_content

    return result, fixes


def _find_closest_method(target: str, available: set) -> Optional[str]:
    """Find the closest matching method name.

    Strategy: prefix match first, then substring match, then suffix match.
    Returns None if no reasonable match found.
    """
    target_lower = target.lower()

    # Exact match (shouldn't happen, but guard)
    if target in available:
        return target

    # Prefix match: target is prefix of available method
    prefix_matches = [m for m in available if m.lower().startswith(target_lower)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    # Substring match: target appears in available method name
    substr_matches = [m for m in available if target_lower in m.lower()]
    if len(substr_matches) == 1:
        return substr_matches[0]

    # Suffix match: target is suffix of available method
    suffix_matches = [m for m in available if m.lower().endswith(target_lower)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    # Word overlap: check if significant words overlap
    target_words = set(re.split(r'[_]', target_lower)) - {'get', 'set', 'is', 'has'}
    best_match = None
    best_overlap = 0
    for method in available:
        if method.startswith("_"):
            continue  # Skip private methods
        method_words = set(re.split(r'[_]', method.lower())) - {'get', 'set', 'is', 'has'}
        overlap = len(target_words & method_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = method

    if best_overlap > 0:
        return best_match

    return None


# =============================================================================
# REQUIREMENTS INFERENCE
# =============================================================================

# Python stdlib module names (3.10+)
_STDLIB_MODULES: Optional[frozenset] = None


def _get_stdlib_modules() -> frozenset:
    """Get stdlib module names, cached."""
    global _STDLIB_MODULES
    if _STDLIB_MODULES is None:
        if hasattr(sys, 'stdlib_module_names'):
            _STDLIB_MODULES = frozenset(sys.stdlib_module_names)
        else:
            # Fallback for Python < 3.10
            _STDLIB_MODULES = frozenset({
                "abc", "argparse", "ast", "asyncio", "base64", "bisect",
                "collections", "configparser", "contextlib", "copy", "csv",
                "dataclasses", "datetime", "decimal", "difflib", "email",
                "enum", "functools", "glob", "hashlib", "hmac", "html",
                "http", "importlib", "inspect", "io", "itertools", "json",
                "logging", "math", "multiprocessing", "operator", "os",
                "pathlib", "pickle", "platform", "pprint", "queue", "random",
                "re", "shutil", "signal", "socket", "sqlite3", "ssl",
                "string", "struct", "subprocess", "sys", "tempfile",
                "textwrap", "threading", "time", "timeit", "traceback",
                "typing", "unittest", "urllib", "uuid", "warnings",
                "weakref", "xml", "zipfile",
            })
    return _STDLIB_MODULES


# Mapping from Python import name → pip package name.
# Only entries where they differ. If not in this table, import name = pip name.
_IMPORT_TO_PIP: Dict[str, str] = {
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "gi": "PyGObject",
    "attr": "attrs",
    "dotenv": "python-dotenv",
    "jose": "python-jose",
    "jwt": "PyJWT",
    "magic": "python-magic",
    "serial": "pyserial",
    "usb": "pyusb",
    "wx": "wxPython",
    "dateutil": "python-dateutil",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "lxml": "lxml",
    "google": "google-api-python-client",
}


def _import_to_pip_name(import_name: str) -> str:
    """Convert Python import name to pip package name."""
    return _IMPORT_TO_PIP.get(import_name, import_name)


def _infer_requirements(all_code: str) -> List[str]:
    """Infer third-party requirements from import statements.

    Scans for import/from statements, filters out stdlib modules.
    Returns pip-installable package names (not import names).
    """
    stdlib = _get_stdlib_modules()

    # Find all import statements
    import_pattern = re.compile(r'^\s*(?:from\s+(\S+)|import\s+(\S+))', re.MULTILINE)
    matches = import_pattern.findall(all_code)

    third_party = set()
    for from_mod, import_mod in matches:
        module = from_mod or import_mod
        # Get top-level module
        top = module.split('.')[0]
        # Skip relative imports and stdlib
        if top.startswith('.') or top.startswith('_'):
            continue
        if top in stdlib:
            continue
        if top:
            third_party.add(_import_to_pip_name(top))

    return sorted(third_party)


def _infer_runtime_requirements(capabilities: Any) -> List[str]:
    """Infer additional pip requirements from RuntimeCapabilities.

    Args:
        capabilities: RuntimeCapabilities instance

    Returns:
        List of pip package names needed for runtime features
    """
    deps: List[str] = []

    if capabilities.has_llm_client:
        deps.append("httpx")

    if capabilities.has_event_loop and capabilities.event_loop_type == "websocket":
        deps.extend(["uvicorn", "fastapi", "websockets"])

    # sqlite3 and subprocess are stdlib — no extra deps
    # asyncio is stdlib — no extra deps

    return deps


# =============================================================================
# FILE GENERATORS
# =============================================================================

def _detect_web_framework(all_code: str) -> Optional[str]:
    """Detect web framework from generated code imports."""
    if 'fastapi' in all_code.lower() or 'from fastapi' in all_code:
        return 'fastapi'
    if 'flask' in all_code.lower() or 'from flask' in all_code:
        return 'flask'
    if 'django' in all_code.lower() or 'from django' in all_code:
        return 'django'
    return None


def _generate_main_py(
    blueprint: Dict[str, Any],
    component_names: List[str],
    all_code: str,
    grouped_files: Optional[Dict[str, Dict[str, str]]] = None,
    entity_types: Optional[frozenset] = None,
    runtime_capabilities: Optional[Any] = None,
) -> str:
    """Generate main.py entry point with imports and component wiring.

    Imports components from source modules and instantiates them
    so main.py actually runs the generated application.

    When runtime_capabilities is set, generates an async entry point that
    initializes the runtime infrastructure (state, LLM, tools) and starts
    the event loop.

    Args:
        blueprint: Compiled blueprint dict
        component_names: List of component names
        all_code: All generated code concatenated
        grouped_files: Optional grouped file mapping
        entity_types: Optional domain-specific entity types (default: software)
        runtime_capabilities: Optional RuntimeCapabilities for async runtime
    """
    # Check if this is a runtime-capable system
    if runtime_capabilities and runtime_capabilities.has_event_loop:
        return _generate_async_main_py(
            blueprint, component_names, grouped_files, entity_types,
            runtime_capabilities,
        )

    ent_types = entity_types if entity_types is not None else _ENTITY_TYPES
    framework = _detect_web_framework(all_code)
    domain = blueprint.get("domain", "application")
    core_need = blueprint.get("core_need", "")

    lines = [
        '"""',
        f'{domain} — entry point.',
        f'{core_need[:80]}' if core_need else '',
        '"""',
        '',
    ]

    # Build imports from grouped source files
    if grouped_files:
        for filename, components in sorted(grouped_files.items()):
            module = filename.replace(".py", "")
            class_names = sorted(_to_pascal(n) for n in components.keys())
            lines.append(f"from {module} import {', '.join(class_names)}")
        lines.append('')

    # Classify components for wiring
    entities = []
    processes = []
    for comp in blueprint.get("components", []):
        name = comp.get("name", "")
        if name in component_names:
            comp_type = comp.get("type", "entity").lower()
            pascal = _to_pascal(name)
            if comp_type in ent_types:
                entities.append(pascal)
            else:
                processes.append(pascal)

    if framework == 'fastapi':
        lines.extend([
            'import uvicorn',
            '',
            '',
            'def main():',
            '    """Start the application."""',
            '    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)',
            '',
        ])
    elif framework == 'flask':
        lines.extend([
            '',
            '',
            'def main():',
            '    """Start the application."""',
            '    app.run(host="0.0.0.0", port=5000, debug=True)',
            '',
        ])
    else:
        lines.extend([
            '',
            '',
            'def main():',
            f'    """Run {domain}."""',
        ])
        # Instantiate components
        if entities or processes:
            for name in entities:
                var = _safe_variable_name(name)
                lines.append(f'    {var} = {name}()')
            for name in processes:
                var = _safe_variable_name(name)
                lines.append(f'    {var} = {name}()')
            lines.append('')
            # Find entry point component — look for CLI, REPL, App, or Main-like components
            entry_var = None
            entry_method = None
            for name in processes + entities:
                name_lower = name.lower()
                var = _safe_variable_name(name)
                if any(kw in name_lower for kw in ('cli', 'repl', 'app', 'application', 'server', 'runner')):
                    entry_var = var
                    # Detect likely entry method from the generated code
                    for method in ('run', 'run_interactive', 'start', 'main', 'execute', 'launch', 'input_loop'):
                        if method in all_code:
                            entry_method = method
                            break
                    if not entry_method:
                        entry_method = 'run'
                    break
            if entry_var and entry_method:
                lines.append(f'    {entry_var}.{entry_method}()')
            else:
                # No obvious entry point — check for sys.argv patterns
                has_argv = 'sys.argv' in all_code or 'argparse' in all_code
                if has_argv and processes:
                    var = _safe_variable_name(processes[0])
                    lines.append(f'    {var}.run()')
                else:
                    lines.append(f'    print("{domain} initialized with {len(entities) + len(processes)} components")')
        else:
            lines.append(f'    print("{domain} is running")')
        lines.append('')

    lines.extend([
        '',
        'if __name__ == "__main__":',
        '    main()',
        '',
    ])

    return "\n".join(lines)


def _generate_async_main_py(
    blueprint: Dict[str, Any],
    component_names: List[str],
    grouped_files: Optional[Dict[str, Dict[str, str]]],
    entity_types: Optional[frozenset],
    runtime_capabilities: Any,
) -> str:
    """Generate async main.py for agent systems with runtime infrastructure."""
    ent_types = entity_types if entity_types is not None else _ENTITY_TYPES
    domain = blueprint.get("domain", "application")
    core_need = blueprint.get("core_need", "")
    port = runtime_capabilities.default_port

    lines = [
        '"""',
        f'{domain} — async runtime entry point.',
        f'{core_need[:80]}' if core_need else '',
        '"""',
        '',
        'import asyncio',
        'import logging',
        '',
    ]

    # Runtime infrastructure imports
    if runtime_capabilities.has_persistent_state:
        lines.append('from state import StateStore')
    if runtime_capabilities.has_llm_client:
        lines.append('from llm_client import LLMClient')
    if runtime_capabilities.has_tool_execution:
        lines.append('from tools import ToolExecutor')
    if runtime_capabilities.has_event_loop:
        lines.append('from runtime import Runtime')
    if runtime_capabilities.can_compile:
        lines.append('from compiler import ToolCompiler')
    if runtime_capabilities.can_share_tools:
        lines.append('from tool_manager import ToolManager')
        lines.append('from motherlabs_platform.instance_identity import InstanceIdentityStore')
    lines.append('from config import Config')
    lines.append('')

    # Component imports from grouped files
    if grouped_files:
        for filename, components in sorted(grouped_files.items()):
            module = filename.replace(".py", "")
            class_names = sorted(_to_pascal(n) for n in components.keys())
            lines.append(f"from {module} import {', '.join(class_names)}")
        lines.append('')

    # Build component lists
    processes = []
    for comp in blueprint.get("components", []):
        name = comp.get("name", "")
        if name in component_names:
            pascal = _to_pascal(name)
            processes.append((name, pascal))

    lines.extend([
        '',
        'async def main():',
        f'    """Start {domain} runtime."""',
        '    logging.basicConfig(level=logging.INFO)',
        '    config = Config()',
        '',
    ])

    # Initialize runtime services
    if runtime_capabilities.has_persistent_state:
        lines.append('    state = StateStore(config.state_path)')
    if runtime_capabilities.has_llm_client:
        lines.append('    llm = LLMClient(config.llm_provider, config.llm_api_key)')
    if runtime_capabilities.has_tool_execution:
        lines.append('    tools = ToolExecutor()')

    # Mother agent services
    corpus_path = runtime_capabilities.corpus_path or "~/motherlabs/corpus.db"
    if runtime_capabilities.can_share_tools:
        lines.append('')
        lines.append('    identity_store = InstanceIdentityStore()')
        lines.append('    identity = identity_store.get_or_create_self()')
        lines.append('    instance_id = identity.instance_id')
    if runtime_capabilities.can_compile:
        lines.append(f'    compiler = ToolCompiler(corpus_path="{corpus_path}")')
    if runtime_capabilities.can_share_tools:
        lines.append(f'    tool_manager = ToolManager(instance_id=instance_id, corpus_path="{corpus_path}")')

    lines.append('')

    # Create runtime and register components
    init_args = []
    if runtime_capabilities.has_persistent_state:
        init_args.append('state=state')
    if runtime_capabilities.has_llm_client:
        init_args.append('llm=llm')
    if runtime_capabilities.has_tool_execution:
        init_args.append('tools=tools')
    if runtime_capabilities.can_compile:
        init_args.append('compiler=compiler')
    if runtime_capabilities.can_share_tools:
        init_args.append('tool_manager=tool_manager')

    lines.append(f'    runtime = Runtime({", ".join(init_args)})')
    lines.append('')
    lines.append('    # Register components')

    # Build constructor args matching runtime contract
    ctor_args = ", ".join(init_args)  # same args as Runtime()

    for original_name, pascal in processes:
        var = _safe_variable_name(pascal)
        lines.append(f'    {var} = {pascal}({ctor_args})')
        lines.append(f'    runtime.register("{original_name}", {var})')

    lines.extend([
        '',
        f'    await runtime.start(port=config.port)',
        '',
        '',
        'if __name__ == "__main__":',
        '    asyncio.run(main())',
        '',
    ])

    return "\n".join(lines)


_RESERVED_VARS = frozenset({
    "state", "llm", "tools", "runtime", "config",
    "logging", "asyncio", "port", "os", "sys", "json",
    "compiler", "tool_manager", "instance_id", "identity", "identity_store",
})


def _safe_variable_name(name: str) -> str:
    """Convert name to safe snake_case variable, avoiding reserved names.

    When _to_snake(name) collides with a runtime infrastructure variable
    (e.g. 'llm', 'state', 'runtime'), appends '_component' suffix.
    """
    snake = _to_snake(name)
    return f"{snake}_component" if snake in _RESERVED_VARS else snake


def _generate_init_py(exports: List[str]) -> str:
    """Generate __init__.py with exports."""
    if not exports:
        return ""
    lines = ['"""Package exports."""', '']
    # Import from submodules — we don't know exact source so just declare __all__
    lines.append(f"__all__ = {exports!r}")
    lines.append('')
    return "\n".join(lines)


def _generate_readme(blueprint: Dict[str, Any], project_name: str) -> str:
    """Generate README.md from blueprint."""
    domain = blueprint.get("domain", project_name)
    core_need = blueprint.get("core_need", "")
    components = blueprint.get("components", [])

    lines = [
        f"# {domain.title()}",
        "",
    ]

    if core_need:
        lines.extend([core_need.strip(), "", ""])

    if components:
        lines.append("## Components")
        lines.append("")
        for comp in components:
            name = comp.get("name", "")
            comp_type = comp.get("type", "")
            desc = comp.get("description", "")
            lines.append(f"- **{name}** ({comp_type}): {desc[:100]}")
        lines.append("")

    lines.extend([
        "## Getting Started",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python main.py",
        "```",
        "",
        "---",
        "*Generated by Motherlabs semantic compiler.*",
        "",
    ])

    return "\n".join(lines)


def _generate_pyproject_toml(project_name: str, requirements: List[str]) -> str:
    """Generate pyproject.toml."""
    deps = ", ".join(f'"{r}"' for r in requirements)
    return f"""[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "{project_name}"
version = "0.1.0"
description = "Generated by Motherlabs semantic compiler"
requires-python = ">=3.10"
dependencies = [{deps}]
"""


def _generate_test_stubs(
    grouped_files: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    """Generate test stub files for each source file."""
    test_files: Dict[str, str] = {}

    for filename, components in grouped_files.items():
        module = filename.replace(".py", "")
        test_filename = f"test_{filename}"

        lines = [
            '"""Tests for ' + module + '."""',
            'import pytest',
            '',
            '',
        ]

        for name in components:
            pascal = _to_pascal(name)
            lines.extend([
                f'class Test{pascal}:',
                f'    """Tests for {name}."""',
                '',
                f'    def test_{_slugify(name)}_exists(self):',
                f'        """Verify {name} can be instantiated."""',
                '        # TODO: Implement',
                '        pass',
                '',
                '',
            ])

        test_files[test_filename] = "\n".join(lines)

    return test_files


# =============================================================================
# SYNTAX VALIDATION
# =============================================================================

def validate_syntax(code: str, filename: str = "<string>") -> Optional[str]:
    """Validate Python code syntax via ast.parse.

    Returns None if valid, error message string if invalid.
    """
    try:
        ast.parse(code, filename=filename)
        return None
    except SyntaxError as e:
        return f"{filename}:{e.lineno}: {e.msg}"


def validate_all_code(
    generated_code: Dict[str, str],
    file_extension: str = ".py",
) -> List[str]:
    """Validate syntax for all generated code.

    Returns list of error messages (empty = all valid).
    Skips ast.parse for non-Python output formats.
    """
    if file_extension != ".py":
        return []  # No Python syntax check for non-Python output
    errors = []
    for name, code in generated_code.items():
        err = validate_syntax(code, filename=f"{name}.py")
        if err:
            errors.append(err)
    return errors


def validate_cross_module(
    resolved_files: Dict[str, str],
    file_extension: str = ".py",
) -> List[str]:
    """Validate cross-module imports resolve to real names.

    For each relative import (from .module import Name), checks that the
    source module actually defines the imported name at top level.

    Returns list of warning strings (not errors — advisory only).
    """
    if file_extension != ".py":
        return []

    # Build module → top-level names mapping
    module_exports: Dict[str, set] = {}
    for filename, content in resolved_files.items():
        module = filename.replace(".py", "")
        try:
            tree = ast.parse(content, filename=filename)
        except SyntaxError:
            continue
        names = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
        module_exports[module] = names

    warnings = []
    for filename, content in resolved_files.items():
        try:
            tree = ast.parse(content, filename=filename)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 1:
                source_module = node.module
                if source_module in module_exports:
                    for alias in node.names:
                        imported_name = alias.name
                        if imported_name not in module_exports[source_module]:
                            warnings.append(
                                f"{filename}: imports '{imported_name}' from .{source_module}, "
                                f"but {source_module}.py does not define it"
                            )
    return warnings


# =============================================================================
# OUTPUT CLEANUP
# =============================================================================

def _clean_project_dir(project_dir: str) -> List[str]:
    """Remove stale generated files from project dir before writing.

    Removes: *.py files, requirements.txt, pyproject.toml, README.md,
    blueprint.json, tests/ directory. Preserves: .git, .env, venv,
    user-created files in non-standard locations.
    """
    removed: List[str] = []
    if not os.path.exists(project_dir):
        return removed

    generated_suffixes = {'.py', '.txt', '.toml', '.md', '.json'}

    for entry in os.listdir(project_dir):
        path = os.path.join(project_dir, entry)
        # Preserve hidden dirs/files (.git, .env, etc.) and venv dirs
        if entry.startswith('.') or entry in ('venv', '.venv', '__pycache__'):
            continue
        if entry == 'tests' and os.path.isdir(path):
            shutil.rmtree(path)
            removed.append('tests/')
        elif os.path.isfile(path):
            _, ext = os.path.splitext(entry)
            if ext in generated_suffixes:
                os.remove(path)
                removed.append(entry)

    return removed


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def write_project(
    generated_code: Dict[str, str],
    blueprint: Dict[str, Any],
    output_dir: str,
    config: Optional[ProjectConfig] = None,
    entity_types: Optional[frozenset] = None,
    file_extension: str = ".py",
) -> ProjectManifest:
    """Write a complete runnable project to disk.

    Args:
        generated_code: Dict[component_name, code_string] from emission
        blueprint: Compiled blueprint dict
        output_dir: Base output directory
        config: Optional ProjectConfig
        entity_types: Optional domain-specific entity types (default: software)
        file_extension: Output file extension (default: ".py")

    Returns:
        ProjectManifest with written files list
    """
    if config is None:
        config = ProjectConfig()

    _is_python = file_extension == ".py"

    # 1. Determine project name
    project_name = config.project_name or _infer_project_name(blueprint)

    # 2. Create project directory
    project_dir = os.path.join(output_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)

    # 2b. Clean stale files from previous builds
    if config.clean_before_write and os.path.exists(project_dir):
        _clean_project_dir(project_dir)

    files_written: List[str] = []
    file_contents: Dict[str, str] = {}
    total_lines = 0

    # 3. Group components into files
    grouped = _group_components(
        generated_code, blueprint, config.per_component_files, entity_types,
        file_extension=file_extension,
    )

    # 4. Resolve cross-file imports (Python only)
    all_component_names = list(generated_code.keys())
    if _is_python:
        resolved = _resolve_imports(grouped, all_component_names)
    else:
        # Non-Python: just concatenate code per file, no import resolution
        resolved = {}
        for filename, components in grouped.items():
            resolved[filename] = "\n\n".join(components.values())

    # 4a. Detect and break circular imports
    if _is_python:
        cycles = _detect_import_cycles(resolved)
        if cycles:
            resolved = _break_import_cycles(resolved, cycles)

    # 4a2. Reconcile interface mismatches
    if _is_python:
        resolved, _reconcile_fixes = _reconcile_interfaces(resolved)

    # 4b. Cross-module validation (advisory)
    cross_warnings = tuple(validate_cross_module(resolved, file_extension))

    # 4c. Sanitize relative imports (convert from .foo to from foo)
    if _is_python:
        for filename in list(resolved.keys()):
            resolved[filename] = re.sub(
                r'^from \.(\w)', r'from \1', resolved[filename], flags=re.MULTILINE,
            )

    # 5. Write source files
    for filename, content in resolved.items():
        filepath = os.path.join(project_dir, filename)
        _write_file(filepath, content)
        files_written.append(filename)
        file_contents[filename] = content
        total_lines += content.count('\n') + 1

    # 6. Write __init__.py (Python only)
    if _is_python:
        exports = [_to_pascal(n) for n in all_component_names]
        init_content = _generate_init_py(exports)
        _write_file(os.path.join(project_dir, "__init__.py"), init_content)
        files_written.append("__init__.py")
        file_contents["__init__.py"] = init_content

    # 7. Write runtime scaffold files (if runtime capabilities present)
    rt_cap = config.runtime_capabilities
    if rt_cap is not None and _is_python:
        from core.runtime_scaffold import (
            generate_runtime_py, generate_state_py, generate_tools_py,
            generate_llm_client_py, generate_config_py, generate_recompile_py,
            generate_compiler_py, generate_tool_manager_py,
        )
        scaffold_files = {
            "runtime.py": generate_runtime_py(rt_cap, blueprint, all_component_names),
            "state.py": generate_state_py(rt_cap, blueprint, all_component_names),
            "tools.py": generate_tools_py(rt_cap, blueprint, all_component_names),
            "llm_client.py": generate_llm_client_py(rt_cap, blueprint, all_component_names),
            "config.py": generate_config_py(rt_cap, blueprint, all_component_names),
            "recompile.py": generate_recompile_py(rt_cap, blueprint, all_component_names),
            "compiler.py": generate_compiler_py(rt_cap, blueprint, all_component_names),
            "tool_manager.py": generate_tool_manager_py(rt_cap, blueprint, all_component_names),
        }
        for fname, code in scaffold_files.items():
            if code:  # empty string = capability disabled
                _write_file(os.path.join(project_dir, fname), code)
                files_written.append(fname)
                file_contents[fname] = code
                total_lines += code.count('\n') + 1

    # 7b. Embed blueprint.json for self-recompilation context
    if rt_cap is not None and rt_cap.has_self_recompile:
        bp_json = json.dumps(blueprint, indent=2, default=str)
        _write_file(os.path.join(project_dir, "blueprint.json"), bp_json)
        files_written.append("blueprint.json")
        file_contents["blueprint.json"] = bp_json

    # 8. Write main.py (Python only)
    entry_point_path = "main.py" if _is_python else ""
    if _is_python and config.entry_point:
        all_code = "\n".join(generated_code.values())
        main_content = _generate_main_py(
            blueprint, all_component_names, all_code, grouped, entity_types,
            runtime_capabilities=rt_cap,
        )
        _write_file(os.path.join(project_dir, "main.py"), main_content)
        files_written.append("main.py")
        file_contents["main.py"] = main_content
        total_lines += main_content.count('\n') + 1

    # 9. Write requirements.txt
    all_code = "\n".join(generated_code.values())
    requirements = _infer_requirements(all_code) if _is_python else []
    # Add runtime-specific dependencies
    if rt_cap is not None and _is_python:
        runtime_deps = _infer_runtime_requirements(rt_cap)
        for dep in runtime_deps:
            if dep not in requirements:
                requirements.append(dep)
        requirements = sorted(requirements)
    req_content = "\n".join(requirements) + "\n" if requirements else ""
    _write_file(os.path.join(project_dir, "requirements.txt"), req_content)
    files_written.append("requirements.txt")
    file_contents["requirements.txt"] = req_content

    # 9. Write pyproject.toml (Python only)
    if _is_python:
        toml_content = _generate_pyproject_toml(project_name, requirements)
        _write_file(os.path.join(project_dir, "pyproject.toml"), toml_content)
        files_written.append("pyproject.toml")
        file_contents["pyproject.toml"] = toml_content

    # 10. Write README.md
    readme_content = _generate_readme(blueprint, project_name)
    _write_file(os.path.join(project_dir, "README.md"), readme_content)
    files_written.append("README.md")
    file_contents["README.md"] = readme_content

    # 11. Write tests (Python only)
    if _is_python and config.tests:
        tests_dir = os.path.join(project_dir, "tests")
        os.makedirs(tests_dir, exist_ok=True)
        _write_file(os.path.join(tests_dir, "__init__.py"), "")
        files_written.append("tests/__init__.py")
        file_contents["tests/__init__.py"] = ""

        test_stubs = _generate_test_stubs(grouped)
        for test_filename, test_content in test_stubs.items():
            _write_file(os.path.join(tests_dir, test_filename), test_content)
            files_written.append(f"tests/{test_filename}")
            file_contents[f"tests/{test_filename}"] = test_content
            total_lines += test_content.count('\n') + 1

    return ProjectManifest(
        project_dir=project_dir,
        files_written=tuple(sorted(files_written)),
        entry_point=entry_point_path,
        total_lines=total_lines,
        file_contents=file_contents,
        cross_module_warnings=cross_warnings,
    )


def _write_file(path: str, content: str) -> None:
    """Write content to file, creating parent dirs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
