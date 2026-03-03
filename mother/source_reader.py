"""
Source reader — AST-based structural analysis of Mother's own codebase.

LEAF module. Stdlib only (ast, os, pathlib, dataclasses, typing, logging).
No imports from core/ or mother/.

Reads Python source files via ast.parse(), extracts structural summaries
(classes, methods, functions, imports, docstrings), and produces compressed
text representations suitable for self-compilation context.
"""

import ast
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("mother.source_reader")

# Directories to skip during package traversal
_SKIP_DIRS = frozenset({
    "tests", "__pycache__", "htmlcov", "archive", ".git",
    ".venv", "node_modules", ".pytest_cache", ".mypy_cache",
})

# Default packages to read
_DEFAULT_PACKAGES = ("core", "kernel", "mother", "agents", "adapters",
                     "cli", "motherlabs_platform")

# Stdlib modules to exclude from import tracking
_STDLIB_PREFIXES = frozenset({
    "os", "sys", "re", "ast", "json", "time", "math", "typing",
    "pathlib", "logging", "hashlib", "sqlite3", "collections",
    "dataclasses", "functools", "itertools", "abc", "enum",
    "threading", "asyncio", "queue", "shutil", "subprocess",
    "importlib", "inspect", "textwrap", "copy", "io", "socket",
    "http", "urllib", "uuid", "datetime", "tempfile", "traceback",
    "contextlib", "string", "struct", "signal", "platform",
    "unittest", "warnings", "weakref", "operator", "statistics",
    "secrets", "base64", "hmac", "csv", "configparser",
    "multiprocessing", "concurrent", "pickle", "gzip", "zipfile",
})

# Max items per extraction
_MAX_METHODS_PER_CLASS = 15
_MAX_FUNCTIONS_PER_MODULE = 20


@dataclass(frozen=True)
class MethodSummary:
    """A method within a class."""
    name: str
    args: str
    return_annotation: str
    is_async: bool


@dataclass(frozen=True)
class ClassSummary:
    """A class definition within a module."""
    name: str
    docstring: str
    bases: Tuple[str, ...]
    methods: Tuple[MethodSummary, ...]
    is_frozen: bool


@dataclass(frozen=True)
class FunctionSummary:
    """A top-level function within a module."""
    name: str
    args: str
    return_annotation: str
    is_async: bool
    docstring: str


@dataclass(frozen=True)
class ModuleSummary:
    """Structural summary of a single Python module."""
    path: str
    docstring: str
    classes: Tuple[ClassSummary, ...]
    functions: Tuple[FunctionSummary, ...]
    imports: Tuple[str, ...]
    line_count: int
    is_leaf: bool


@dataclass(frozen=True)
class SourceSnapshot:
    """Structural snapshot of the entire codebase."""
    project_root: str
    timestamp: float
    modules: Tuple[ModuleSummary, ...]
    total_lines: int
    total_classes: int
    total_functions: int


def _format_args(node: ast.FunctionDef) -> str:
    """Extract argument signature as string."""
    parts = []
    args = node.args
    # Regular args (skip 'self'/'cls')
    for arg in args.args:
        name = arg.arg
        if name in ("self", "cls"):
            continue
        ann = ""
        if arg.annotation:
            ann = f": {ast.unparse(arg.annotation)}"
        parts.append(f"{name}{ann}")
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    if args.kwonlyargs:
        for kw in args.kwonlyargs:
            ann = ""
            if kw.annotation:
                ann = f": {ast.unparse(kw.annotation)}"
            parts.append(f"{kw.arg}{ann}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def _format_return(node: ast.FunctionDef) -> str:
    """Extract return annotation as string."""
    if node.returns:
        return ast.unparse(node.returns)
    return ""


def _get_docstring(node: ast.AST) -> str:
    """Extract first 200 chars of docstring."""
    doc = ast.get_docstring(node)
    if doc:
        return doc[:200]
    return ""


def _is_internal_import(name: str) -> bool:
    """Check if an import is from the project (not stdlib/third-party)."""
    top = name.split(".")[0]
    if top in _STDLIB_PREFIXES:
        return False
    # Known project packages
    if top in ("core", "kernel", "mother", "agents", "adapters",
               "cli", "motherlabs_platform"):
        return True
    return False


def _extract_imports(tree: ast.Module) -> Tuple[str, ...]:
    """Extract non-stdlib imports from module AST."""
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_internal_import(alias.name):
                    imports.add(alias.name.split(".")[0] + "." + ".".join(alias.name.split(".")[1:2]))
        elif isinstance(node, ast.ImportFrom):
            if node.module and _is_internal_import(node.module):
                # Track at module level, not individual names
                imports.add(node.module)
    return tuple(sorted(imports))


def _extract_methods(cls_node: ast.ClassDef) -> Tuple[MethodSummary, ...]:
    """Extract public methods + __init__ from class, capped at _MAX_METHODS_PER_CLASS."""
    methods = []
    for node in cls_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        # Skip private methods except __init__
        if name.startswith("_") and name != "__init__":
            continue
        methods.append(MethodSummary(
            name=name,
            args=_format_args(node),
            return_annotation=_format_return(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
        ))
        if len(methods) >= _MAX_METHODS_PER_CLASS:
            break
    return tuple(methods)


def _detect_frozen(cls_node: ast.ClassDef) -> bool:
    """Detect @dataclass(frozen=True) decorator."""
    for dec in cls_node.decorator_list:
        if isinstance(dec, ast.Call):
            func = dec.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "dataclass":
                for kw in dec.keywords:
                    if kw.arg == "frozen":
                        if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            return True
    return False


def read_module(file_path: str, project_root: str) -> Optional[ModuleSummary]:
    """Parse a single Python file and extract structural summary.

    Returns None on SyntaxError or empty files.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except (OSError, IOError) as e:
        logger.debug(f"Cannot read {file_path}: {e}")
        return None

    if not source.strip():
        return None

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        logger.debug(f"SyntaxError in {file_path}: {e}")
        return None

    line_count = source.count("\n") + 1
    docstring = _get_docstring(tree)
    is_leaf = "LEAF" in (docstring or "")

    # Extract classes
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(ClassSummary(
                name=node.name,
                docstring=_get_docstring(node),
                bases=tuple(
                    ast.unparse(b) for b in node.bases
                ),
                methods=_extract_methods(node),
                is_frozen=_detect_frozen(node),
            ))

    # Extract top-level functions (skip private)
    functions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            functions.append(FunctionSummary(
                name=node.name,
                args=_format_args(node),
                return_annotation=_format_return(node),
                is_async=isinstance(node, ast.AsyncFunctionDef),
                docstring=_get_docstring(node)[:100] if _get_docstring(node) else "",
            ))
            if len(functions) >= _MAX_FUNCTIONS_PER_MODULE:
                break

    # Relative path from project root
    rel_path = os.path.relpath(file_path, project_root)

    return ModuleSummary(
        path=rel_path,
        docstring=docstring,
        classes=tuple(classes),
        functions=tuple(functions),
        imports=_extract_imports(tree),
        line_count=line_count,
        is_leaf=is_leaf,
    )


def read_package(
    package_dir: str,
    project_root: str,
    skip_dirs: Optional[frozenset] = None,
) -> List[ModuleSummary]:
    """Walk a package directory and read all Python modules.

    Skips tests/, __pycache__/, and other non-source directories.
    Returns modules sorted by path for deterministic output.
    """
    skip = skip_dirs or _SKIP_DIRS
    modules = []
    for dirpath, dirnames, filenames in os.walk(package_dir):
        # Filter out skip directories (mutate dirnames in-place)
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            mod = read_module(fpath, project_root)
            if mod is not None:
                modules.append(mod)
    modules.sort(key=lambda m: m.path)
    return modules


def read_codebase(
    project_root: str,
    packages: Tuple[str, ...] = _DEFAULT_PACKAGES,
) -> SourceSnapshot:
    """Read all packages in the codebase and produce a structural snapshot.

    Args:
        project_root: Absolute path to project root.
        packages: Package directory names to scan.

    Returns:
        SourceSnapshot with all modules and aggregate stats.
    """
    all_modules = []
    root = Path(project_root)
    for pkg in packages:
        pkg_dir = root / pkg
        if pkg_dir.is_dir():
            all_modules.extend(read_package(str(pkg_dir), project_root))

    total_lines = sum(m.line_count for m in all_modules)
    total_classes = sum(len(m.classes) for m in all_modules)
    total_functions = sum(len(m.functions) for m in all_modules)

    return SourceSnapshot(
        project_root=project_root,
        timestamp=time.time(),
        modules=tuple(all_modules),
        total_lines=total_lines,
        total_classes=total_classes,
        total_functions=total_functions,
    )


def format_source_summary(snapshot: SourceSnapshot, max_words: int = 5000) -> str:
    """Compress a SourceSnapshot into a text summary.

    Large files (>500 lines) get 3-4 lines of detail.
    Small LEAFs get 1 line. Groups by package.
    Includes dependency graph at the end.

    Args:
        snapshot: The source snapshot to format.
        max_words: Budget for output text length.

    Returns:
        Compressed structural summary as text.
    """
    lines = []
    lines.append(f"# Source Structure ({len(snapshot.modules)} modules, "
                 f"{snapshot.total_lines:,} lines, {snapshot.total_classes} classes, "
                 f"{snapshot.total_functions} functions)")
    lines.append("")

    # Group by package
    packages: dict = {}
    for mod in snapshot.modules:
        pkg = mod.path.split(os.sep)[0] if os.sep in mod.path else mod.path.split("/")[0]
        packages.setdefault(pkg, []).append(mod)

    word_count = 20  # header

    for pkg_name in sorted(packages.keys()):
        mods = packages[pkg_name]
        lines.append(f"## {pkg_name}/ ({len(mods)} files, "
                     f"{sum(m.line_count for m in mods):,} lines)")
        word_count += 10

        for mod in mods:
            if word_count >= max_words:
                lines.append(f"  ... ({len(mods) - mods.index(mod)} more files truncated)")
                break

            leaf_tag = " [LEAF]" if mod.is_leaf else ""
            # Large files get more detail
            if mod.line_count > 500:
                lines.append(f"  {mod.path} ({mod.line_count} lines){leaf_tag}")
                word_count += 8
                for cls in mod.classes[:5]:
                    frozen_tag = " (frozen)" if cls.is_frozen else ""
                    method_names = [m.name for m in cls.methods[:8]]
                    lines.append(f"    class {cls.name}{frozen_tag}: "
                                 f"{', '.join(method_names)}")
                    word_count += len(method_names) + 4
                for fn in mod.functions[:5]:
                    async_tag = "async " if fn.is_async else ""
                    ret = f" → {fn.return_annotation}" if fn.return_annotation else ""
                    lines.append(f"    {async_tag}def {fn.name}({fn.args}){ret}")
                    word_count += 8
            elif mod.line_count > 100:
                # Medium files: class names + key functions
                cls_names = [c.name for c in mod.classes]
                fn_names = [f.name for f in mod.functions[:5]]
                parts = []
                if cls_names:
                    parts.append(f"classes: {', '.join(cls_names)}")
                if fn_names:
                    parts.append(f"fns: {', '.join(fn_names)}")
                detail = "; ".join(parts) if parts else ""
                lines.append(f"  {mod.path} ({mod.line_count} lines){leaf_tag}"
                             f"{' — ' + detail if detail else ''}")
                word_count += 10 + len(cls_names) + len(fn_names)
            else:
                # Small files: one line
                cls_names = [c.name for c in mod.classes]
                detail = f" — {', '.join(cls_names)}" if cls_names else ""
                lines.append(f"  {mod.path} ({mod.line_count} lines){leaf_tag}{detail}")
                word_count += 6

        lines.append("")

    # Append dependency graph if budget remains
    if word_count < max_words - 200:
        dep_graph = format_dependency_graph(snapshot)
        if dep_graph:
            lines.append(dep_graph)

    return "\n".join(lines)


def format_dependency_graph(snapshot: SourceSnapshot) -> str:
    """Format cross-package import edges.

    Only includes edges between different packages.
    Format: `source_pkg.module → target_pkg`
    """
    edges: dict = {}  # source_module → set of target packages
    for mod in snapshot.modules:
        src_pkg = mod.path.split(os.sep)[0] if os.sep in mod.path else mod.path.split("/")[0]
        for imp in mod.imports:
            tgt_pkg = imp.split(".")[0]
            if tgt_pkg != src_pkg:
                edges.setdefault(mod.path, set()).add(tgt_pkg)

    if not edges:
        return ""

    lines = ["## Cross-Package Dependencies"]
    for src_path in sorted(edges.keys()):
        targets = sorted(edges[src_path])
        lines.append(f"  {src_path} → {', '.join(targets)}")
    return "\n".join(lines)


def source_snapshot_to_facts(snapshot: SourceSnapshot) -> List[dict]:
    """Convert a SourceSnapshot to KnowledgeFact-compatible dicts.

    Produces facts with deterministic fact_ids keyed on module path
    (idempotent on re-analysis). Categories: pattern (architectural),
    tool (dependencies), capability (what modules provide).

    Returns list of dicts matching KnowledgeFact fields.
    """
    now = time.time()
    facts = []

    # Architectural facts: one per module
    for mod in snapshot.modules:
        cls_names = [c.name for c in mod.classes]
        fn_names = [f.name for f in mod.functions[:5]]

        value_parts = []
        if cls_names:
            value_parts.append(f"classes: {', '.join(cls_names)}")
        if fn_names:
            value_parts.append(f"functions: {', '.join(fn_names)}")
        if mod.is_leaf:
            value_parts.append("LEAF module")
        value_parts.append(f"{mod.line_count} lines")

        facts.append({
            "fact_id": f"source:module:{mod.path}",
            "category": "pattern",
            "subject": mod.path,
            "predicate": "contains",
            "value": "; ".join(value_parts),
            "confidence": 0.95,
            "source": "source_reader:ast",
            "first_seen": now,
            "last_confirmed": now,
            "access_count": 0,
        })

    # Capability facts: one per class with methods
    for mod in snapshot.modules:
        for cls in mod.classes:
            if not cls.methods:
                continue
            method_names = [m.name for m in cls.methods if m.name != "__init__"]
            if not method_names:
                continue
            facts.append({
                "fact_id": f"source:class:{mod.path}:{cls.name}",
                "category": "capability",
                "subject": cls.name,
                "predicate": "provides",
                "value": ", ".join(method_names[:10]),
                "confidence": 0.95,
                "source": "source_reader:ast",
                "first_seen": now,
                "last_confirmed": now,
                "access_count": 0,
            })

    # Cross-package dependency facts
    for mod in snapshot.modules:
        src_pkg = mod.path.split(os.sep)[0] if os.sep in mod.path else mod.path.split("/")[0]
        cross_deps = set()
        for imp in mod.imports:
            tgt_pkg = imp.split(".")[0]
            if tgt_pkg != src_pkg:
                cross_deps.add(tgt_pkg)
        if cross_deps:
            facts.append({
                "fact_id": f"source:deps:{mod.path}",
                "category": "tool",
                "subject": mod.path,
                "predicate": "depends on",
                "value": ", ".join(sorted(cross_deps)),
                "confidence": 0.95,
                "source": "source_reader:ast",
                "first_seen": now,
                "last_confirmed": now,
                "access_count": 0,
            })

    return facts
